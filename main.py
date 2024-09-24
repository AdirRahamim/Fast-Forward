import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import os
from datasets import load_from_disk
import argparse
import random
import numpy as np
from time_recorder import TimeRecorder
from utils import print_trainable_parameters
from peft import LoraConfig, get_peft_model
from flops_profiler.profiler import FlopsProfiler
import concurrent.futures
from utils import INSTRUCTION_KEY, END_KEY, RESPONSE_KEY_NL, DEFAULT_CHAT_TEMPLATE
from utils import get_medical_dataset, get_instruction_dataset, get_chat_dataset
from trl import DataCollatorForCompletionOnlyLM

if torch.cuda.is_available():
    from accelerate import Accelerator
    from accelerate.utils import set_seed


def finetune(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.tokenizer_max_length)
    tokenizer.pad_token = tokenizer.eos_token

    if args.task == 'instruction':
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]}
        )
    elif args.task == 'chat':
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    if not os.path.isdir(os.path.join(args.output_dir, "tokenized_dataset")):
        if args.task == 'medical':
            tokenized_datasets = get_medical_dataset(args, tokenizer)
            train_partition = 'train'
            test_partition = 'test'
        elif args.task == 'instruction':
            tokenized_datasets = get_instruction_dataset(args, tokenizer)
            train_partition = 'train'
            test_partition = 'test'
        elif args.task == 'chat':
            tokenized_datasets = get_chat_dataset(args, tokenizer)
            train_partition = 'train_sft'
            test_partition = 'test_sft'
    else:
        tokenized_datasets = load_from_disk(os.path.join(args.output_dir, "tokenized_dataset"))


    # Model
    if 'llama' in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if 'pythia' in args.model_name:
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["query_key_value"],
            lora_dropout=0.1,
            bias="none",
            use_dora=args.use_dora
        )
    else:
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            use_dora=args.use_dora
        )
    lora_model = get_peft_model(model, config)
    print_trainable_parameters(lora_model)

    optimizer = AdamW(lora_model.parameters(), lr=args.lr)

    tokenized_datasets.set_format("torch")
    if args.task == 'instruction':
        data_collator = DataCollatorForCompletionOnlyLM(RESPONSE_KEY_NL, tokenizer=tokenizer, mlm=False,
                                                        return_tensors="pt",
                                                        pad_to_multiple_of=8)
        model.resize_token_embeddings(len(tokenizer))
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(tokenized_datasets[train_partition], batch_size=args.batch_size, shuffle=True,
                                  collate_fn=data_collator)

    predictor_dataset = tokenized_datasets[test_partition].select(range(32))
    tokenized_datasets[test_partition] = tokenized_datasets[test_partition].select(range(32, len(tokenized_datasets[test_partition])))
    predictor_dataloader = DataLoader(predictor_dataset, batch_size=args.eval_batch_size, collate_fn=data_collator)
    val_dataloader = DataLoader(tokenized_datasets[test_partition], batch_size=args.eval_batch_size, collate_fn=data_collator)

    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation_steps)

    lora_model, optimizer, train_dataloader, val_dataloader, predictor_dataloader = accelerator.prepare(
        lora_model, optimizer, train_dataloader, val_dataloader, predictor_dataloader
    )

    def evaluate(dataloader=val_dataloader):
        with torch.no_grad():
            lora_model.eval()
            losses = []
            for step, batch in enumerate(dataloader):
                outputs = lora_model(batch["input_ids"], attention_mask=batch['attention_mask'],
                                     labels=batch["labels"])

                losses.append(accelerator.gather(outputs.loss).unsqueeze(0))
            loss = torch.mean(torch.cat(losses))
            perplexity = torch.exp(loss)
            return loss.item(), perplexity.item()

    num_epochs = 5
    training_min_loss = None

    if args.fast_forward:
        predictor_steps = 0
        next_predictor_step = 6
        save_every = accelerator.gradient_accumulation_steps

        param_dict = {}
        param_dict_empty = {}
        for name, _ in lora_model.named_parameters():
            if 'lora' in name:
                param_dict[name] = []
                param_dict_empty[name] = []

    runtime_recorder = TimeRecorder()
    if args.flops_profiler:
        prof = FlopsProfiler(model)
    total_flops = 0
    param_update_flops = 0
    fast_forward_test_flops = 0

    for epoch in range(num_epochs):
        lora_model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_model):
                runtime_recorder.train_start()
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                if args.flops_profiler:
                    prof.start_profile()
                outputs = lora_model(input_ids, attention_mask=attention_mask, labels=batch["labels"])
                if args.flops_profiler:
                    prof.stop_profile()
                    total_flops += prof.get_total_flops()
                    prof.end_profile()
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                runtime_recorder.train_end()

            if batch_idx % args.evaluate_every == 0 and not (epoch == 0 and batch_idx < args.num_stabilization_steps):
                total_loss, ppl = evaluate()
                if args.flops_profiler:
                    accelerator.print(
                        f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, "
                        f"test Loss: {total_loss}, test ppl: {ppl}, total flops: {total_flops / 1e9} e9")
                else:
                    accelerator.print(
                        f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, "
                        f"test Loss: {total_loss}, test ppl: {ppl}")

                if training_min_loss is None or total_loss < training_min_loss:
                    training_min_loss = total_loss

            if args.fast_forward and not (epoch == 0 and batch_idx < args.num_stabilization_steps) and batch_idx % save_every == 0:
                runtime_recorder.train_predictor_start()
                predictor_steps += 1
                if predictor_steps == next_predictor_step or predictor_steps == next_predictor_step-1:
                    model_state_dict = lora_model.state_dict()
                    for key in param_dict.keys():
                        param_dict[key].append(model_state_dict[key].detach().clone())
                        if (predictor_steps == next_predictor_step):
                            param_dict[key] = torch.stack(param_dict[key])

                runtime_recorder.train_predictor_end()

            if args.fast_forward and predictor_steps == next_predictor_step:
                runtime_recorder.train_predictor_start()
                keys = list(param_dict.keys())
                def calc_diff(param_name):
                    return param_name, param_dict[param_name][-1] - param_dict[param_name][-2]

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = executor.map(calc_diff, keys)

                difference_dict = {param_name: value for param_name, value in results}

                prev_loss = None
                fast_forward_step = 0
                runtime_recorder.train_predictor_end()

                while True:
                    runtime_recorder.train_predictor_start()
                    model_dict = lora_model.state_dict()
                    def cal_update(param_name):
                        return param_name, model_dict[param_name] + difference_dict[param_name]

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        results = executor.map(cal_update, keys)

                    param_dict.update(results)
                    lora_model.load_state_dict(param_dict, strict=False)
                    runtime_recorder.train_predictor_end()

                    param_update_flops += sum(torch.numel(difference_dict[param_name]) for param_name in keys)
                    lora_model.eval()

                    ## TODO REMOVE AFTER TEST
                    total_loss, ppl = evaluate()
                    accelerator.print(
                        f"Real eval step {fast_forward_step}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {total_loss}, ppl: {ppl}")
                    ## UNTIL HERE

                    fast_forward_step += 1
                    if args.flops_profiler:
                        prof.start_profile()
                    runtime_recorder.train_predictor_start()
                    total_loss, ppl = evaluate(dataloader=predictor_dataloader)
                    runtime_recorder.train_predictor_end()
                    if args.flops_profiler:
                        prof.stop_profile()
                        fast_forward_test_flops += prof.get_total_flops()
                        prof.end_profile()
                    accelerator.print(
                        f"Fast Forward step: {fast_forward_step}, Loss: {total_loss}, ppl: {ppl}"
                        f" test flops: {fast_forward_test_flops / 1e9} e9, model param update flops: {param_update_flops / 1e9} e9")

                    lora_model.train()
                    if prev_loss is not None and total_loss > prev_loss:
                        break
                    prev_loss = total_loss

                next_predictor_step = predictor_steps + 6
                param_dict.update(param_dict_empty)

    time_stats = runtime_recorder.get_time_stats()
    if args.flops_profiler:
        accelerator.print(f'Total train flops: {3*total_flops / 1e9} e9')
        accelerator.print(f'Total fast forward test flops: {fast_forward_test_flops / 1e9} e9')
        accelerator.print(f'Total fast forward param update flops: {param_update_flops / 1e9} e9')

    return time_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='chat', choices=['medical', 'instruction', 'chat'])
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B',
                        choices=['EleutherAI/pythia-1.4b', 'EleutherAI/pythia-2.8b', 'EleutherAI/pythia-6.9b',
                                 "meta-llama/Meta-Llama-3-8B"])

    # Model training arguments
    parser.add_argument('--lr', type=float, default=2.0e-05)
    parser.add_argument('--accumulation_steps', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--tokenizer_max_length', type=int, default=1024)
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=16)

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--evaluate_every', type=int, default=512)

    # Fast Forward arguments
    parser.add_argument('--num_stabilization_steps', type=int, default=50)
    parser.add_argument('--flops_profiler', type=bool, default=True, help='Enable flops profiler')
    parser.add_argument('--fast_forward', type=bool, default=True,
                        help='Regular training or training with fast forward')
    parser.add_argument('--use_dora', type=bool, default=False, help='Use DoRA or LoRA training')

    args = parser.parse_args()
    print(args)
    os.makedirs('args.task', exist_ok=True)
    args.output_dir = os.path.join(args.task, args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    set_seed(args.seed)
    time_stats = finetune(args)
    print(f'Time stats: {time_stats}')