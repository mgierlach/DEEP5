import os
import torch
import torchvision
from torchvision.datasets.utils import download_url
import zipfile

train_path = 'train'

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import imageio
import multiprocessing
import numpy as np
import torch


class Images(Dataset):
    def __init__(self):
        """
        Initialize dataset.
        Here we can define transformations for the dataset aswell.
        """
        self.labels = pd.read_csv("./train/labels.csv")
        self.normalize = transforms.Normalize((0, 0, 0), (1, 1, 1))
        self.to_tensor = transforms.ToTensor()

        self.transformations = transforms.Compose(
            [self.to_tensor, self.normalize])

    def __getitem__(self, index):
        index = index.item() + 1  #indexing starts from 1 in our dataset
        try:
            img = np.array(
                imageio.imread(
                    './train/images/im' + str(index) + '.jpg', pilmode="RGB"))
        except:
            raise

        img = self.transformations(img)
        # e.g. data = self.center_crop(data)
        label = self.labels[self.labels["img"] == index].drop(
            "img", axis="columns")
        label = torch.from_numpy(np.array(label))
        return (img, label.squeeze())

    def __len__(self):
        return len(self.labels)


batch_size = 64
imgs = Images()

# Create 0.8, 0.1, 0.1 splits for training, validation, testing
train_size = int(0.8 * len(imgs))
validation_size = len(imgs) - train_size

train_dataset, validation_dataset = torch.utils.data.random_split(
    imgs, [train_size, validation_size])

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=10)
valid_loader = DataLoader(
    dataset=validation_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    print('Using gpu.')
    device = torch.device('cuda')
else:
    print('Using CPU.')
    device = torch.device('cpu')



class Net(nn.Module):
    def __init__(self, output_size=14, batch_size=32):
        super(Net, self).__init__()
        self.output_size = output_size
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=20, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=7)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.fc1 = nn.Linear(100*18*18, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.out = nn.Linear(512, output_size)

    def forward(self, x):
        # Convolutional layer expects the input to be of shape (batch_size, channel_dim, x_dim, y_dim)
        conv = F.relu(self.conv1(x))
       	pooled = self.pool(conv)

        conv2 = F.relu(self.conv2(pooled))
       	pooled2 = self.pool2(conv2)

        conv3 = F.relu(self.conv3(pooled2))
        conv4 = F.relu(self.conv4(conv3))
        pooled3 = self.pool3(conv4)

        conv5 = F.relu(self.conv5(pooled3))
        pooled4 = self.pool4(conv5)

        flatten = pooled4.view(-1, 100*18*18)

       	h = F.relu(self.fc1(flatten))
        h = F.relu(self.fc2(h))
        output = self.out(h)
       	return output




model = Net(batch_size=batch_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0002)
# I think Binary Cross Entropy is OK for multilabel. LogitsLoss is just added sigmoid with log-sum-exp trick
criterion = nn.BCEWithLogitsLoss().to(device)


def train(model, loader, optimizer, criterion):

    epoch_loss = 0

    model.train()

    for idx, (X, y) in enumerate(loader):
        X = X.to(device)
        y = y.to(device)
        if idx % 100 == 0:
            print(f"batch {idx} out of {len(loader)}")

        optimizer.zero_grad()

        predictions = model(X)

        loss = criterion(predictions, y.float())

        loss.backward()

        optimizer.step()

        epoch_loss += loss.data.item()

    return epoch_loss / len(loader)


from sklearn.metrics import f1_score


def evaluate(model, loader, criterion):

    epoch_loss = 0
    epoch_f1 = 0

    model.eval()  # disables normalizations like dropout, batchnorm etc..

    with torch.no_grad():  # disables autograd engine

        for idx, (X, y) in enumerate(loader):
            X = X.to(device)
            y = y.to(device)
            predictions = model(X)
            if (idx==0):
       	       	print(np.around(torch.sigmoid(predictions[0:5]).cpu().numpy(), decimals=2))
                print(y[0:5].cpu().numpy())



            loss = criterion(predictions, y.float())
            # just round for label true/false
            rounded_preds = torch.round(
                torch.sigmoid(predictions)
            )  # also sigmoid because its not used without loss function

            # micro-averaged f1 is used for determining the best model.
            f1 = f1_score(rounded_preds.cpu(), y.cpu(), average='micro')

            epoch_loss += loss.data.item()
            epoch_f1 += f1.item()

    return epoch_loss / len(loader), epoch_f1 / len(loader)


N_EPOCHS = 70
training_losses = []
validation_losses = []

for epoch in range(N_EPOCHS):
    print(f"started training epoch {epoch + 1}..")
    train_loss = train(model, train_loader, optimizer, criterion)
    print(f"started validating epoch {epoch + 1}..")
    valid_loss, valid_f1 = evaluate(model, valid_loader, criterion)
    print(f'Training epoch {epoch +1},  loss: {train_loss}')
    print(
        f'Validation epoch {epoch +1}: loss: {valid_loss}, F1 score: {valid_f1}'
    )
    # Store all the losses per epoch for making a plot later on
    training_losses = training_losses + [train_loss]
    validation_losses = validation_losses + [valid_loss]

print("Training losses:", training_losses)
print("Validation losses:", validation_losses)
