import torch
import torch.onnx
from CNN import ConvNeuralNet  # assuming your model is in 'model_file.py'

# Create a sample input matching the input size of the model (batch size, channels, height, width)
dummy_input = torch.randn(1, 1, 48, 48)  # Adjust (1, 1, 28, 28) if your input has different dimensions

# Instantiate the model and set it to evaluation mode
model = ConvNeuralNet(num_classes=7)  # adjust num_classes to your use case
model.eval()

# Export the model to an ONNX file
onnx_file_path = "conv_neural_net.onnx"
torch.onnx.export(model, dummy_input, onnx_file_path, export_params=True, opset_version=10,
                  do_constant_folding=True, input_names=['input'], output_names=['output'])

print(f"Model exported to {onnx_file_path}")
