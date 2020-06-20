import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from MNIST import Net

plt.style.use('grayscale')




    
classes = ['0','1','2','3','4','5','6','7','8','9']


model = Net()
#model.cuda()


model.load_state_dict(torch.load('C:/Users/walle/Desktop/Portfolio/Projekty/AM/model_MNIST.pt'))


def am(model, digit, Lambda = 0.01, Alfa = 0.1, accuracy = 0.001, max_iters = 100000):
    
    model.eval()
    x_next = torch.FloatTensor(1, 1, 28, 28).uniform_(-1, 1) #random tensor with dimentions [Batch, channels, H, W]
    x_next.cuda()
    L = Lambda
    A = Alfa
    acc = accuracy
    Max = max_iters 
    
    for i in range(Max):
        
        x_current = torch.autograd.Variable(x_next,requires_grad = True) 
       
        out = model(x_current)[0]
        z = -1*torch.log10(out[digit]) + L*(torch.norm(x_current.view(-1), p=2))**2
        z.backward(retain_graph = True)
        df = x_current.grad[0][0]
        x_next = x_current - A*df
        step = x_next - x_current
        step = step.view(-1,)

        model.zero_grad()
        if torch.norm(step, p =2) <= acc:
            print('{}-stop'.format(digit))
            break
        
        if i == Max-1:
            print(torch.norm(step, p=2))
    
    return(x_next)
    
    
    

def imshow_MNIST(x):
    x = x.detach() 
    if torch.max(x) <=1 and torch.min(x) >=-1: 
        x = x/2 + 0.5
        x = x.view(28,28)
        plt.imshow(x)
        
    else:
        x = (x + (0 - torch.min(x))) / (torch.max(x) + (0 - torch.min(x))) 
        x = x.view(28,28)
        plt.imshow(x)
        
    
    
def am_vanilla_obrazki(x):
    
    f = open("wyniki dla lambda = {}".format(x), "w+")
    for i in range(len(classes)):   
        im = am(model, i, Lambda = x)
        imshow_MNIST(im)
        plt.savefig('{}-{}.png'.format(classes[i], x))
        print('{} - zrobione'.format(classes[i]))
        f.write("wynik dla {}: {} \r\n".format(i, model(im)))
    f.close()

def am_relu(model, digit, Lambda = 0.01, Alfa = 0.1, accuracy = 0.001, max_iters = 100000):

    model.eval()
    x_next = torch.FloatTensor(1, 1, 28, 28).uniform_(0, 1) 

    L = Lambda
    A = Alfa
    acc = accuracy
    Max = max_iters 
    
    for i in range(Max):
        
        x_current = torch.autograd.Variable(x_next,requires_grad = True) 

        out = model(x_current)[0]
        z = -1*torch.log10(out[digit]) + L*(torch.norm(x_current.view(-1), p=2))**2
        z.backward(retain_graph = True)
        df = x_current.grad[0][0]
        x_next = x_current - A*df
        x_next = F.relu(x_next)
        step = x_next - x_current
        step = step.view(-1,)

        model.zero_grad()
        if torch.norm(step, p =2) <= acc:
            print('{}-stop'.format(digit))
            break
        
        if i == Max-1:
            print(torch.norm(step, p=2))
    
    return(x_next)
    
def imshow_relu(x):
    x = x.detach()
    if torch.max(x) > 1:
        x = x/torch.max(x)
    x = x.view(28,28)
    plt.imshow(x)
        
    
def am_relu_obrazki(x):
    f = open("wyniki relu dla lambda = {}".format(x), "w+")
    for i in range(len(classes)):   
        im = am_relu(model, i, Lambda = x)
        imshow_relu(im)
        plt.savefig('relu {}-{}.png'.format(classes[i], x))
        print('{} - zrobione'.format(classes[i]))
        f.write("wynik dla {}: {} \r\n".format(i, model(im)))
    f.close()


am_relu_obrazki(0.01)
