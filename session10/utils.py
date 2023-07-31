from random import randint
from typing import Any, Union

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
from torchvision import transforms
from tqdm import tqdm


def plot_sampledata(loader):
    batch_data, batch_label = next(iter(loader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].permute(1, 2, 0).numpy())
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

    plt.show()


def GetCorrectPredCount(pPrediction: torch, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


class Trainer:
    def __init__(self, model: nn.Module, device: torch.device, optimizer, scheduler=None) -> None:
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_criterion = nn.CrossEntropyLoss()
        self.test_criterion = nn.CrossEntropyLoss(reduction="sum")

        self.train_acc = []
        self.train_losses = []
        self.test_acc = []
        self.test_losses = []

    def train(self, train_loader: DataLoader):
        self.model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = self.train_criterion(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            correct += GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
            )

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(train_loader))

    def evaluate(self, test_loader: DataLoader):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += self.test_criterion(output, target).item()  # sum up batch loss

                correct += GetCorrectPredCount(output, target)

        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100.0 * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    def plot_history(self):
        """
        Plot the training and test accuracy, loss, and epochs.

        This function plots the training and test accuracy, loss, and epochs of a neural network model. It takes no parameters and has no return value.

        The function first calculates the maximum training accuracy and its corresponding epoch number. It then calculates the maximum test accuracy and its corresponding epoch number. The function prints a table showing the maximum accuracy at each epoch for both the training and test sets.

        The function then plots two subplots: one for the loss vs epoch and one for the accuracy vs epoch. The subplot for the loss vs epoch shows the training loss and test loss over the epochs. The subplot for the accuracy vs epoch shows the training accuracy and test accuracy over the epochs.

        Example usage:
        model = NeuralNetwork()
        model.train()
        model.plot_history()
        """

        max_train = max(self.train_acc)
        ep_train = self.train_acc.index(max_train) + 1

        max_test = max(self.test_acc)
        ep_test = self.test_acc.index(max_test) + 1
        print("Set\t Max Acc@Epoch\t Last Epoch Acc")
        print(f"train\t {max_train:0.2f}@{ep_train}\t\t{self.train_acc[-1]:0.2f}")
        print(f"test\t {max_test:0.2f}@{ep_test}\t\t{self.test_acc[-1]:0.2f}")

        # For loss and epochs
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label="Training Loss")  # plotting the training loss
        plt.plot(self.test_losses, label="Test Loss")  # plotting the testing loss
        # putting the labels on plot
        plt.title("Loss vs Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid()
        plt.legend()

        # For accuracy and epochs
        plt.subplot(1, 2, 2)
        plt.plot(self.train_acc, label="Training Accuracy")  # plotting the training accuracy
        plt.plot(self.test_acc, label="Test Accuracy")  # plotting the testing accuracy
        # putting the labels in plot
        plt.title("Accuracy vs Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.grid()
        plt.legend()

        plt.show()


def find_lr(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    criterion: Any,
    dataloader: DataLoader,
):
    """
    Finds the learning rate for a given model using the LR Finder technique.

    Args:
        model (nn.Module): The model to find the learning rate for.
        device (torch.device): The device to run the model on.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        criterion (Any): The loss function used for training the model.
        dataloader (DataLoader): The data loader used for training the model.
    """

    # Create an instance of the LR Finder
    lr_finder = LRFinder(model, optimizer, criterion, device=device)

    # Run the range test to find the optimal learning rate
    lr_finder.range_test(dataloader, end_lr=10, num_iter=200, step_mode="exp")

    # Plot the loss-learning rate graph for inspection
    lr_finder.plot()

    # Reset the model and optimizer to their initial state
    lr_finder.reset()


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    cols, rows = 4, 6
    figure = plt.figure(figsize=(20, 20))
    for i in range(1, cols * rows + 1):
        k = np.random.randint(0, len(loader.dataset))  # random points from test dataset

        img, label = loader.dataset[k]  # separate the image and label
        img = img.unsqueeze(0)  # adding one dimention
        pred = model(img.to(device))  # Prediction

        figure.add_subplot(rows, cols, i)  # making the figure
        plt.title(f"Predcited label {pred.argmax().item()}\n True Label: {label}")  # title of plot
        plt.axis("off")  # hiding the axis
        plt.imshow(img.squeeze(), cmap="gray")  # showing the plot

    plt.show()


def plot_misclassified(
    model: Any,
    data_loader: DataLoader,
    device: torch.device,
    transformations: A.Compose,
    title: str = "Misclassified (pred/ truth)",
):
    count = 1
    no_misclf: int = 10
    rows, cols = 2, int(no_misclf / 2)
    figure = plt.figure(figsize=(cols * 3, rows * 3))

    classes = data_loader.dataset.classes
    dataset = data_loader.dataset.ds

    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        while True:
            k = randint(0, len(dataset))
            img, label = dataset[k]
            img = np.array(img)

            aug_img = transformations(image=img)["image"]
            pred = model(aug_img.unsqueeze(0).to(device)).argmax().item()  # Prediction
            if pred != label:
                figure.add_subplot(rows, cols, count)  # adding sub plot
                plt.title(f"{classes[pred]} / {classes[label]}")  # title of plot
                plt.axis("off")
                plt.imshow(img)

                count += 1
                if count == no_misclf + 1:
                    break

    plt.suptitle(title, fontsize=15)
    plt.show()


def per_class_accuracy(model: Any, device: torch.device, data_loader: DataLoader):
    model = model.to(device)
    model.eval()

    classes = data_loader.dataset.classes
    nc = len(classes)
    class_correct = list(0.0 for i in range(nc))
    class_total = list(0.0 for i in range(nc))
    with torch.inference_mode():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(nc):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("[x] Accuracy of ::")
    for i in range(nc):
        print("\t[*] %8s : %2d %%" % (classes[i], 100 * class_correct[i] / class_total[i]))
