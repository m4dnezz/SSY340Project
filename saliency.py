import torch
import torchvision.transforms as transforms
from CNN import ConvNeuralNet
from matplotlib import pyplot as plt
import cv2
import numpy as np

num_classes = 7
model = ConvNeuralNet(num_classes)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()


# Load and preprocess the image
def load_and_preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))  # Resize to match model input
    img = transforms.ToTensor()(img)  # Convert to Tensor
    img = img.unsqueeze(0)  # Add batch dimension
    return img


# Generate saliency map
def generate_saliency_map(model, img_tensor):
    img_tensor.requires_grad_()  # Enable gradient tracking
    output = model(img_tensor)  # Forward pass
    output_idx = output.argmax()  # Get index of the top class
    output[0, output_idx].backward()  # Backpropagate to get gradients
    saliency, _ = torch.max(img_tensor.grad.data.abs(), dim=1)  # Get the maximum gradient
    saliency = saliency.squeeze().numpy()
    saliency = np.where(saliency < 0.2, 0, saliency)  # Thresholding for visualization
    return saliency


# Visualize the image and saliency map
def plot_results(img_paths, img_tensors, model):
    num_images = len(img_paths)
    plt.figure(figsize=(15, 5))

    for i in range(num_images):
        original_image = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)  # Read original image
        original_image = cv2.resize(original_image, (48, 48))  # Ensure size matches
        saliency_map = generate_saliency_map(model, img_tensors[i])

        # Plot Original Image
        plt.subplot(1, num_images, i + 1)
        plt.imshow(original_image, cmap='gray')
        plt.axis('off')

        # Plot Saliency Map)
        plt.imshow(saliency_map, cmap='inferno', alpha=0.5)
        plt.axis('off')
        plt.colorbar()

    plt.tight_layout()
    plt.show()


# Main execution
img_paths = [
    "FER2013/test/angry/PrivateTest_7622844.jpg",
    "FER2013/test/disgust/PublicTest_67735286.jpg",
    "FER2013/test/fear/PrivateTest_8831137.jpg"
    "FER2013/test/happy/PrivateTest_4518933.jpg"
    "FER2013/test/neutral/PrivateTest_11752870.jpg",
    "FER2013/test/sad/PrivateTest_55014751.jpg",
    "FER2013/test/surprise/PrivateTest_14592510.jpg"
]

# Load all images into tensors
img_tensors = [load_and_preprocess_image(path) for path in img_paths]

# Call the plot function
plot_results(img_paths, img_tensors, model)
