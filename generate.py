import os
from medmnist import OCTMNIST
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


# Define the directory to save the dataset
base_dir = 'OCTMNIST_data'
transform = transforms.Compose([
    transforms.ToTensor(),
])
# Create directories for each class
print("Generating fodler 'OCTMNIST_data'")
classes = ['0', '1', '2', '3']
for cls in classes:
    os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

# Download the dataset
print("Downloading 'OCTMNIST' data")
dataset = OCTMNIST(split='train', download=True, transform=transform)

# Get images and labels
images = np.array(dataset.imgs)
labels = dataset.labels

# Save images to corresponding class folders
for index, (image, label) in enumerate(zip(images, labels)):
    if index == 99:
        break
    img = Image.fromarray(image, mode='L')  # Convert numpy array to PIL Image
    assert img.size == (28, 28)
    label_str = str(label[0])
    img.save(os.path.join(base_dir, label_str, f'{index}_label_{label[0]}.jpg'), 'JPEG')

print("Downloaded 100 Retinal OCT images saved to respective class folders.")
