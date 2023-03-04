import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt




def lect_1():
    x = torch.Tensor([5,2])
    y = torch.Tensor([1,3])
    return x+y

def download_mnist_data():
    train = datasets.MNIST("",train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("",train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    return train, test

def check_balance_data(trainset):
    counter_data = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    total = 0
    for data in trainset:
        Xs, ys = data
        for i in ys:
            counter_data[int(i)] += 1
            total += 1
    return counter_data, total

def lect_2(train, test):
    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)
    counter_data, total = check_balance_data(trainset)
    print(counter_data)
    for i in counter_data:
        print(f"{i} : {counter_data[i]/total*100}")
    return trainset, testset


print(lect_1())
train, test = download_mnist_data()
trainset, testset = lect_2(train,test)




