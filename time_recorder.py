import time

import torch


class TimeRecorder(object):

    def __init__(self):
        self.total_train_time = 0
        self.total_predictor_time = 0
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda:
            self.train_start_time = torch.cuda.Event(enable_timing=True)
            self.train_end_time = torch.cuda.Event(enable_timing=True)
            self.train_predictor_start_time = torch.cuda.Event(enable_timing=True)
            self.train_predictor_end_time = torch.cuda.Event(enable_timing=True)

    def train_start(self):
        if self.is_cuda:
            self.train_start_time.record()
        else:
            self.train_start_time = time.time()

    def train_end(self):
        if torch.cuda.is_available():
            self.train_end_time.record()
            torch.cuda.synchronize()
            self.total_train_time += self.train_start_time.elapsed_time(self.train_end_time) / 1000.0
        else:
            self.train_end_time = time.time()
            self.total_train_time += self.train_end_time - self.train_start_time

    def train_predictor_start(self):
        if self.is_cuda:
            self.train_predictor_start_time.record()
        else:
            self.train_predictor_start_time = time.time()

    def train_predictor_end(self):
        if torch.cuda.is_available():
            self.train_predictor_end_time.record()
            torch.cuda.synchronize()
            self.total_predictor_time += self.train_predictor_start_time.elapsed_time(self.train_predictor_end_time) / 1000.0
        else:
            self.train_predictor_end_time = time.time()
            self.total_predictor_time += self.train_predictor_end_time - self.train_predictor_start_time

    def get_time_stats(self):
        return {
            "training_time": self.total_train_time,
            "training_time_predictor": self.total_predictor_time
        }