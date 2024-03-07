import csv
import os
import random
from functools import partialmethod
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
class AverageMeter(object):
 """Computes and stores the average and current value"""
 def __init__(self):
 self.reset()
 def reset(self):
 self.val = 0
 self.avg = 0
 self.sum = 0
 self.count = 0
 def update(self, val, n=1):
 self.val = val
 self.sum += val * n
 self.count += n
 self.avg = self.sum / self.count
class Logger(object):
 def __init__(self, infile, header):

 self.log_file = open(infile,'w+', newline = '')
 #self.log_file = infile
 self.logger = csv.writer(self.log_file, delimiter=',')
 self.logger.writerow(header)
 print(header)
 self.header = header
 def __del(self):
 self.log_file.close()
 def log(self, values):
 write_values = []
 for col in self.header:
 assert col in values

 write_values.append(values[col])
 #print(write_values)
 self.logger.writerow(write_values)
 self.log_file.flush()
def calculate_accuracy(outputs, targets):
 with torch.no_grad():
 #THIS HAS BEEN CHANGED SUBSTANTIALLY
 batch_size = targets.size(0)
 if batch_size > 1:
 _, pred = outputs.topk(1, 1, largest=True, sorted=True)
 pred = pred.t()
 _, label = targets.topk(1, 1, largest=True, sorted=True)
 label = label.t()
 #correct = pred.eq(targets.view(1, -1))
 correct = pred.eq(label)
 #print(pred)
 #print(label)
 n_correct_elems = correct.float().sum().item()
 return n_correct_elems / batch_size
 else:
 _, pred = torch.max(outputs,1)
 _, predt = torch.max(targets,1)
 #print('Guess '+str(pred.item())+' Label '+str(predt.item()))
 n_correct_elems = 0
 if(pred == predt):
 n_correct_elems = n_correct_elems+1
 return n_correct_elems / batch_size
def calculate_precision_and_recall(outputs, targets, pos_label=1):
 with torch.no_grad():
 #print(outputs)
 _, pred = outputs.topk(1, 1, largest=True, sorted=True) #used to be sorted is true
 precision, recall, _, _ = precision_recall_fscore_support(
 targets.view(-1, 1).cpu().numpy(),
 pred.cpu().numpy())
 return precision[pos_label], recall[pos_label]
def worker_init_fn(worker_id):
 torch_seed = torch.initial_seed()
 random.seed(torch_seed + worker_id)
 if torch_seed >= 2**32:
 torch_seed = torch_seed % 2**32
 np.random.seed(torch_seed + worker_id)
def get_lr(optimizer):
 lrs = []
 for param_group in optimizer.param_groups:
 lr = float(param_group['lr'])
 lrs.append(lr)
 return max(lrs)
def partialclass(cls, *args, **kwargs):
 class PartialClass(cls):
 __init__ = partialmethod(cls.__init__, *args, **kwargs)
 return PartialClass 
