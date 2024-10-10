import torch
import numpy as np
import random
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
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

# Data setup
train_path = "./FER2013/train"
test_path = "./FER2013/test"

batch_size = 32
my_trans = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])
workers = 6

# large set
train_set = FER2013(train_path, transform=my_trans)
test_set = FER2013(test_path, transform=my_trans)
train_dataloader = DataLoader(train_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
test_dataloader = DataLoader(test_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)

# Small set, sanity check. 98% train acc with this
# sample_size = 1000  # Set this to the size of your subset
# random_indices = random.sample(range(len(train_set)), sample_size)

# small_train_set = Subset(train_set, random_indices)
# small_test_set = Subset(test_set, random.sample(range(len(test_set)), sample_size))
# train_dataloader = DataLoader(small_train_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
# test_dataloader = DataLoader(small_test_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)

# Model setup
num_epochs = 50  # For testing
num_classes = 7
model = ConvNeuralNet(num_classes)

# Training setup
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # remember to use model.to(device), data.to(device)
    print(f"Starting to train model, using {device} for training")
    model, train_losses, train_accs, val_losses, val_accs = training_loop(model, optimizer, loss_fn, train_dataloader, test_dataloader, num_epochs, print_every=100)


# TODO: Fix plotting to analyze results / training
# TODO: Save model and data after each run
# I get like 90% train acc but val acc wont go over 60%....