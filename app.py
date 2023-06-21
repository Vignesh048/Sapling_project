import torch
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import requests

st.header('Sapling Classification 🌱🌱')

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a consistent size
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image data
])

device = torch.device("cpu")

# Pretrained ResNet-18 model
model = models.resnet18()
num_features = model.fc.in_features

# Freeze pretrained weights
for param in model.parameters():
    param.requires_grad = False

# Add two more fully connected layers
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)
)  # Binary classification, output layer with 1 neuron

# Load the saved model
model_path = "sapling.pth"  # Path to the saved model
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define the class labels
class_labels = ['Not a Sapling', 'Sapling']

# Create a dropdown box to select an image file
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

# Check if an image file is uploaded
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)

    # Display the selected image
    st.image(image, caption='Uploaded Image')

    # Display image size
    image_size = image.size
    st.write('Image Size:', image_size)

    # Check if the 'Classify' button is clicked
    if st.button('Classify'):
        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0)

        # Move input tensor to CPU
        input_tensor = input_tensor.to(device)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = (output >= 0.5).int()
            predicted_label = class_labels[predicted_idx.item()]

        # Set the text color based on the prediction result
        if predicted_idx == 1:
            result_color = 'green'
        else:
            result_color = 'red'

        # Show the classification result with color
        st.markdown('<span style="font-size: 48px">Classification Result: </span>'+'<span style="font-size: 48px; color:{}">{}</span>'.format(result_color, predicted_label), unsafe_allow_html=True)
# f'<span style="font-size: 24px, "color:{}";">{classification_result}</span>'