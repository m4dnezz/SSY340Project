import torch
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from PIL import Image
from torch.utils.data import Dataset
from path import Path
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from CNN import ConvNeuralNet
from DataSet import FER2013
from torchvision.datasets import ImageFolder
from functions import train_epoch, training_loop, output_to_label

## Data setup
train_path = "./FER2013/train"
test_path = "./FER2013/test"

batch_size = 2096
my_trans = transforms.Compose([
    transforms.Resize((48, 48)),  
    transforms.ToTensor(),        
])
train_dataloader = DataLoader(FER2013(train_path, transform=my_trans),batch_size,shuffle=True,num_workers=4, pin_memory=True, prefetch_factor=2)
test_dataloader = DataLoader(FER2013(test_path, transform=my_trans),batch_size,shuffle=True,num_workers=4, pin_memory=True, prefetch_factor=2)

## Model setup
num_epochs = 1 # For testing
num_classes = 8
model = ConvNeuralNet(num_classes)

## Training setup
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #remember to use model.to(device), data.to(device)
    print(f"Starting to train model, using {device} for training")
    training_loop(model, optimizer, loss_fn, train_dataloader, test_dataloader, num_epochs, print_every=1)