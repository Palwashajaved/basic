import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. Define the transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),       
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])
 
# 2. Load the dataset
trainset = ImageFolder(root=r'C:\Users\palwa\Desktop\basic\train', transform=transform)
testset = ImageFolder(root=r'C:\Users\palwa\Desktop\basic\test', transform=transform)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# 3. Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(64 * 64 * 3, 128)
        self.fc2 = nn.Linear(128, 2)            # Output size is 2 (pen or pencil)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = self.fc2(x)              # Output logits for 2 classes
        return x

# 4. Instantiate the network, loss function, and optimizer
net = SimpleNN()
criterion = nn.CrossEntropyLoss()  # Suitable for classification
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 5. Train the model
for epoch in range(5):  # Train for 5 epochs
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()  # Zero the parameter gradients

        outputs = net(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

# 6. Test the model
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)  # Get predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

# Save the trained model
torch.save(net.state_dict(), 'model.pth')

# Load the trained model
net = SimpleNN()
net.load_state_dict(torch.load('model.pth'))
net.eval()  # Set the model to evaluation mode

# 7. Classify and display a specific image
def classify_and_display_image(image_path):
    # Open and transform the image
    img = Image.open(image_path)
    transformed_img = transform(img).unsqueeze(0)  # Add batch dimension

    # Classify the image
    with torch.no_grad():
        output = net(transformed_img)
        _, predicted = torch.max(output, 1)

    # Define the class labels
    class_labels = ['pen', 'pencil']
    label = class_labels[predicted.item()]

    # Display the image with the classification result as the title
    plt.imshow(np.asarray(img))  # Display the image
    plt.title(f'The image is classified as: {label}')  # Display the classification result
    plt.axis('off')  # Hide axis
    plt.show()

# 8. Provide the path to an image you want to classify
image_path = r'C:\Users\palwa\Desktop\basic\test\pen\pen-test.jpeg'  # Replace with your image path
classify_and_display_image(image_path)
