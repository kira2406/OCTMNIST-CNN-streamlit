import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cnnModel import imageCNN
from torchvision import transforms
import torch

def load_model():
    model = imageCNN()
    model = torch.load('kushwant_assignment0_part_3.pth')
    model.eval()
    return model

cnn_model = load_model()

def preprocess(img):
    transform = transforms.Compose([
    transforms.ToTensor(),
    ])
    image_p = transform(img)
    image_p = image_p.unsqueeze(0) # Converting shape of tensor from [1, 28, 28] to [1, 1, 28, 28]
    return image_p

def make_prediction(image):
    predictions = []
    print("image", image.shape)
    outputs = cnn_model(image)
    _, predicted = torch.max(outputs.data, 1)
    predictions.extend(predicted.cpu().numpy())
    predictions = np.array(predictions)
    print("predicted", predictions)
    labels = ['Normal (Class 0)', 'Choroidal Neovascularization (CNV) (Class 1)', 'Diabetic Macular Edema (DME) (Class 2)', 'Drusen (Class 3)']
    return labels[predictions[0]]

st.title('OCT MNIST - Retinal OCT Classifier')
st.text('OCT: Optical Coherence Tomography')
img_upload = st.file_uploader(label="Upload Retinal OCT image Here:", type=["png","jpg","jpeg"])


if img_upload:
    img = Image.open(img_upload)
    tensor_image = preprocess(img)

    prediction = make_prediction(tensor_image)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    plt.imshow(img, cmap='gray')
    plt.xticks([],[])
    plt.yticks([],[])
    ax.spines[["top","bottom","right"]].set_visible(False)

    st.pyplot(fig, use_container_width=True)
    st.subheader("Prediction:")
    st.header(prediction)