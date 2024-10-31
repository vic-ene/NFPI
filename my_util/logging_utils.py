import torch 
import os 
import numpy as np
import pandas as pd


class AverageMeter(object):
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GPUTimer():
    def __init__(self, active=True):
        self.active=active

        if torch.cuda.is_available():
            self.time_start = torch.cuda.Event(enable_timing=True)
            self.time_end = torch.cuda.Event(enable_timing=True)

    def start(self):
        if torch.cuda.is_available():
            if self.active:
                self.time_start.record()

    def stop(self, msg):
        if torch.cuda.is_available():
            if self.active:
                self.time_end.record()
                torch.cuda.synchronize()
                print(msg, self.time_start.elapsed_time(self.time_end) / 1000)

