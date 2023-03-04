import torch
import torchvision
from torchvision import datasets, transforms

def lect_1():
    x = torch.Tensor([5,2])
    y = torch.Tensor([1,3])
    return x+y

print(lect_1())