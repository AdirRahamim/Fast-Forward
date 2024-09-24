import os
from datasets import load_dataset

INSTRUCTION_KEY = "### Instruction:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"

DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def get_medical_dataset(args, tokenizer):
    dataset = load_dataset("epfl-llm/guidelines")['train']

    def tokenize_function(examples):
        return tokenizer(examples["clean_text"], truncation=True, max_length=tokenizer.model_max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True,
                                     remove_columns=['id', 'source', 'title', 'clean_text', 'raw_text', 'url',
                                                     'overview'])
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=1032)
    tokenized_datasets.save_to_disk(os.path.join(args.output_dir, "tokenized_dataset"))
    return tokenized_datasets


def get_instruction_dataset(args, tokenizer):
    INTRO_BLURB = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    )
    PROMPT = """{intro}
                       {instruction_key}
                       {instruction}
                       {response_key}
                       {response}
                       {end_key}""".format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction="{instruction}",
        response_key=RESPONSE_KEY,
        response="{response}",
        end_key=END_KEY,
    )

    dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")['train']

    def _add_text(rec):
        instruction = rec["instruction"]
        response = rec["response"]
        rec["text"] = PROMPT.format(
            instruction=instruction,
            response=response)
        return rec

    dataset = dataset.map(_add_text)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.tokenizer_max_length)

    tokenized_datasets = dataset.map(tokenize_function, batched=True,
                                     remove_columns=["instruction", "response", "text"])
    tokenized_datasets = tokenized_datasets.filter(lambda rec: len(rec["input_ids"]) < args.tokenizer_max_length)
    tokenized_datasets = tokenized_datasets.train_test_split(test_size=1000)
    tokenized_datasets.save_to_disk(os.path.join(args.output_dir, "tokenized_datasets"))
    return tokenized_datasets


def get_chat_dataset(args, tokenizer):
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

    def apply_chat_template(example):
        messages = example["messages"]
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        return example

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=tokenizer.model_max_length)

    processed_dataset = dataset.map(apply_chat_template)
    tokenized_datasets = processed_dataset.map(tokenize_function, batched=True,
                                              remove_columns=['prompt', 'prompt_id', 'messages', "text"])
    tokenized_datasets["test_sft"] = tokenized_datasets["test_sft"].select(list(range(1032)))
    tokenized_datasets.save_to_disk(os.path.join(args.output_dir, "tokenized_datasets"))
    return tokenized_datasets