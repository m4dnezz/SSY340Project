import torch
import numpy as np
import torchvision.transforms as transforms
# import torchvision.models as models
from torch.utils.data import DataLoader
from CNN import ConvNeuralNet
from DataSet import FER2013
from functions import training_loop, train_val_plots, confmatrix, image_predictions
from matplotlib import pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Data setup
train_path = "./FER2013/train"
test_path = "./FER2013/test"

batch_size = 128

train_trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert to single-channel grayscale
    transforms.Resize((224, 224)),  # Resize to ensure uniform size
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally
    transforms.RandomRotation(degrees=10),  # Rotate within ±10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight translation
    # transforms.RandomCrop(48, padding=4),  # Crop with padding
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Adjust brightness and contrast
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale
])

val_trans = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
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

num_epochs = 100  # For testing
num_classes = 7
model = ConvNeuralNet(num_classes)
# model = models.resnet18()
# num_feat = model.fc.in_features
# model.fc = torch.nn.Linear(num_feat, 7)
# model.load_state_dict(torch.load("best_model.pth"))


# Training setup
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=2e-4)
train_labels = [label for _, label in train_set]
labels = np.unique(train_labels)
cw = compute_class_weight(class_weight="balanced", classes=labels, y=train_labels)
cw_tens = torch.tensor(cw, dtype=torch.float32).to(torch.device("cuda"))
loss_fn = torch.nn.CrossEntropyLoss(weight=cw_tens)

if __name__ == '__main__':
    print(f"length of train set: {len(train_set)}")
    print(f"length of test set: {len(test_set)}")
    model, train_losses, train_accs, val_losses, val_accs, epochs = training_loop(model, optimizer, loss_fn, train_dataloader,
                                                                                  test_dataloader, num_epochs, print_every=100,
                                                                                  patience=50)
    # model.load_state_dict(torch.load("best_model.pth"))
    # model.eval()
    train_val_plots(train_losses, train_accs, val_losses, val_accs, epochs)
    confmatrix(model, test_dataloader)
    image_predictions(model, test_set, numberofimages=5)
    torch.save(
     {
        "model_state_dict": model.state_dict(),
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
     },
     "./res.ckpt",
    )
    plt.show()
