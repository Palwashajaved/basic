# Simple Image Classification with PyTorch:

This project demonstrates a simple image classification model using PyTorch. The model is designed to classify images of two classes: pen and pencil. The code includes data loading, model training, testing, and single-image classification.

# Project Overview:

This project uses a simple feed-forward neural network (fully connected layers) to classify images into two categories: pen and pencil. It leverages the PyTorch framework for training the model, and the dataset is organized using ImageFolder for easy loading.

# Requirements:

To run this project, you will need the following:

Python 3.x

PyTorch

Torchvision

Pillow (PIL)

Matplotlib

# Model Architecture:

The model used is a simple feed-forward neural network (fully connected layers):

Input layer: Flattens the image from 64x64x3 to a vector.

Hidden layer: Fully connected layer with ReLU activation.

Output layer: Two output nodes representing the two classes (pen and pencil).

# Training:

The model is trained using the following configuration:

Optimizer: Stochastic Gradient Descent (SGD) with momentum.

Loss Function: CrossEntropyLoss (suitable for classification tasks).

Learning Rate: 0.01

Batch Size: 32

Epochs: 5

During training, the model parameters are updated based on the training dataset, and the loss is reported after each epoch.

# Testing and Evaluation:

The model's performance is evaluated using the test dataset. Accuracy is calculated as the percentage of correctly classified images.


# Classifying a Single Image;

To classify a specific image, use the classify_and_display_image() function. This function takes the path of an image, transforms it, and uses the trained model to predict its class. The image is displayed along with the predicted label.

Example usage:
The image is classified as: pen

# Saving and Loading the Model;

The trained model is saved as model.pth. To load a saved model, use the following code:

net.load_state_dict(torch.load('model.pth'))

>Ensure the model architecture is the same when loading the saved model.

# Results:
After training the model for 5 epochs, the achieved accuracy on the test dataset was X%. The model performs well on distinguishing between pens and pencils, and can be improved further with more data or fine-tuning the architecture.

