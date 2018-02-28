from __future__ import division
from util import *
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
import torch
import sys
import os
import wave
import sklearn
import random


a = torch.FloatTensor([[1,2,3],[4,5,6]])
w = np.array([1, 2])
w = torch.from_numpy(w).type(torch.FloatTensor)
ww = torch.diag(w)
print a
print ww
print torch.mm(ww, a)



# print a*w
