import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from CNN import ConvNeuralNet
from DataSet import FER2013
from functions import training_loop, train_val_plots, confmatrix, image_predictions
from matplotlib import pyplot as plt

# Data setup
train_path = "./FER2013/train"
test_path = "./FER2013/test"

batch_size = 64

train_trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to single-channel grayscale
    transforms.Resize((48, 48)),  # Resize to ensure uniform size
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally
    transforms.RandomRotation(degrees=10),  # Rotate within ±10 degrees
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Slight translation
    transforms.RandomCrop(48, padding=4),  # Crop with padding
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Adjust brightness and contrast
    transforms.RandomPerspective(distortion_scale=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale
])

val_trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

workers = 6

# large set
train_set = FER2013(train_path, transform=train_trans)
test_set = FER2013(test_path, transform=val_trans)
train_dataloader = DataLoader(train_set, batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True, prefetch_factor=2, persistent_workers=True)
test_dataloader = DataLoader(test_set, batch_size, shuffle=True, num_workers=workers,
                             pin_memory=True, prefetch_factor=2, persistent_workers=True)

# Small set, sanity check. 98% train acc with this
# sample_size = 100  # Set this to the size of your subset
# random_indices = random.sample(range(len(train_set)), sample_size)

# small_train_set = Subset(train_set, random_indices)
# small_test_set = Subset(test_set, random.sample(range(len(test_set)), sample_size))
# train_dataloader = DataLoader(small_train_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
# test_dataloader = DataLoader(small_test_set, batch_size, shuffle=True, num_workers=workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)

# Model setup
num_epochs = 500  # For testing
num_classes = 7
model = ConvNeuralNet(num_classes)

# Training setup
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    print(f"length of train set: {len(train_set)}")
    print(f"length of test set: {len(test_set)}")
    model, train_losses, train_accs, val_losses, val_accs = training_loop(model, optimizer, loss_fn, train_dataloader,
                                                                          test_dataloader, num_epochs, print_every=1000,
                                                                          patience=50)
    train_val_plots(train_losses, train_accs, val_losses, val_accs, num_epochs)
    confmatrix(model, test_dataloader)
    image_predictions(model, test_set, numberofimages=5)
    plt.show()

# TODO: Save model and data after each run
# TODO: Visualize class imbalance (there is some)
# TODO: Weight classes so we learn from classes with less data
# TODO: Doesnt save model until 50 epochs
# I get like 90% train acc but val acc wont go over 60%....
