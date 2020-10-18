import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from models import *
import numpy as np

device = torch.device('cuda')
print('Using device:', device)
print('GPU:', torch.cuda.get_device_name(0))
net = CNN().to(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

num_epochs = 20
losses = []
batches = len(trainloader)

def train():
   print("TRAINING")
   for epoch in range(num_epochs):

      progress = tqdm(enumerate(trainloader), desc="Loss: ", total=batches)

      total_loss = 0
      for i, (inputs, labels) in progress:
         
         inputs, labels = inputs.to(device), labels.to(device)

         optimizer.zero_grad()

         output= net(inputs)
         loss = criterion(output, labels)
         
         loss.backward()
         optimizer.step()
         
         current_loss = loss.item()
         total_loss += current_loss

         progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))

      losses.append(total_loss/batches)
      print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/batches}")

   torch.save(net, './save')


def test():
   total_correct = 0
   total_images = 0
   confusion_matrix = np.zeros([10,10], int)
   with torch.no_grad():
       for inputs, labels in testloader:
           inputs, labels = inputs.to(device), labels.to(device)
           outputs = net(inputs)
           _, predicted = torch.max(outputs.data, 1)
           total_images += labels.size(0)
           total_correct += (predicted == labels).sum().item()
           for i, l in enumerate(labels):
               confusion_matrix[l.item(), predicted[i].item()] += 1 

   model_accuracy = total_correct / total_images * 100
   print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))

train()
test()