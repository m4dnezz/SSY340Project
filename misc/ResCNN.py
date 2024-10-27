import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        # Identity mapping: Ensure input and output dimensions match.
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.identity = nn.Sequential()

    def forward(self, x):
        identity = self.identity(x)

        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)

        # Add skip connection (identity) to the output
        out += identity
        out = self.relu(out)
        return out


class ConvNeuralNetRes(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNetRes, self).__init__()

        # First Residual Block
        self.residual_block1 = ResidualBlock(in_channels=1, out_channels=32)

        # Second Residual Block
        self.residual_block2 = ResidualBlock(in_channels=32, out_channels=64)

        # Third Residual Block
        self.residual_block3 = ResidualBlock(in_channels=64, out_channels=128)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 6 * 6, 256)  # Flattened size is 128 * 6 * 6 = 4608
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, num_classes)

        # Max-pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # First Residual Block
        out = self.residual_block1(x)
        out = self.max_pool(out)

        # Second Residual Block
        out = self.residual_block2(out)
        out = self.max_pool(out)

        # Third Residual Block
        out = self.residual_block3(out)
        out = self.max_pool(out)

        # Flatten the output from the convolutional layers
        out = out.view(out.size(0), -1)  # Flatten from (batch_size, 128, 6, 6) to (batch_size, 4608)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
