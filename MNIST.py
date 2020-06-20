
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    device = torch.device("cuda:0")
    print('CUDA is available!  Training on GPU ...')
    
    
classes = ['0','1','2','3','4','5','6','7','8','9']

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle = True)

testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform,)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, drop_last=True)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #
        self.conv1 = nn.Conv2d(1, 7, kernel_size = 3, padding = 1) #28x28x1 -> 14X14X7
        self.bn1 = nn.BatchNorm2d(7)
        nn.init.xavier_uniform_(self.conv1.weight)
        
        self.conv2 = nn.Conv2d(7,14, kernel_size = 3, padding = 1)#14X14X7 -> 7X7X14 
        self.bn2 = nn.BatchNorm2d(14)
        nn.init.xavier_uniform_(self.conv2.weight)
        #
        self.pool = nn.MaxPool2d(2,2)# dzieli przez 2
        #
        self.fc1 = nn.Linear(14 * 7 * 7, 100)
        nn.init.xavier_uniform_(self.fc1.weight)
        
        self.fc2 = nn.Linear(100,10)
        nn.init.xavier_uniform_(self.fc2.weight)
        #
        self.dropout = nn.Dropout(0.2)
        
    def forward(self,x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        #
        x = x.view(-1, 14*7*7)
        #
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        #
        x = self.dropout(x)
        x = F.softmax(self.fc2(x), dim =1)
        #
        return(x)
        
        
model = Net()
model.cuda()
print(model)
        




criterion = nn.CrossEntropyLoss()


optimizer = optim.SGD(model.parameters(), lr=0.01)


n_epochs = 30

valid_loss_min = np.Inf # we track changes in  Validation loss, so we start with infinity

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in trainloader:
        
        data, target = data.cuda(), target.cuda()
        

        optimizer.zero_grad()
        

        output = model(data)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()*data.size(0)
        
    ######################    
    # validate the model #
    ######################
    model.eval()
    for data, target in validloader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss 
        valid_loss += loss.item()*data.size(0)
    
    # calculate average losses
    train_loss = train_loss/len(trainloader.dataset)
    valid_loss = valid_loss/len(validloader.dataset)
        
    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_MNIST.pt')
        valid_loss_min = valid_loss


model.load_state_dict(torch.load('.../model_MNIST.pt'))

test_loss = 0.0
class_correct = list(0. for i in range(10))
print(class_correct)
class_total = list(0. for i in range(10))
print(class_total)


for data, target in testloader:
    data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)

    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    
    _, pred = torch.max(output, 1)   
   
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    for i in range(64):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
    # calculate test accuracy for each object class
 # calculate test accuracy for each object class

test_loss = test_loss/len(testloader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
