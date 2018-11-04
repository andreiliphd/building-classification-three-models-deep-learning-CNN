import torch
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split

def loadImages(path):
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        loadedImages.append(plt.imread(path + image))
    return np.array(loadedImages)

panel = loadImages('./photo_small/panel/') / 255
modern = loadImages('./photo_small/modern/') / 255
photo_up = np.concatenate((panel, modern), axis=0)
photo = photo_up.swapaxes(3, 1).swapaxes(3,2)
label = np.concatenate((np.zeros(20), np.ones(20)), axis=0)
X_train, X_test, y_train, y_test = train_test_split(photo, label, test_size=0.1, random_state=42, shuffle=False)
X_train_torch = torch.from_numpy(X_train).float()
X_test_torch = torch.from_numpy(X_test).float()
y_train_torch = torch.from_numpy(y_train).long()
y_test_torch = torch.from_numpy(y_test).long()

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=(3, 3)),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, kernel_size=(3, 3)),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=(2, 2)),
        torch.nn.Dropout(0.25),
        Flatten(),
        torch.nn.Linear(457856, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 2)
        )

loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(60):
    y_pred = model(X_train_torch)
    loss = loss_fn(y_pred, y_train_torch)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def accuracy_calc(X, y):
    correct = 0
    total_correct = 0
    with torch.no_grad():
        outputs = model(X).detach().numpy()
        label = y.detach().numpy()
        for number in range(outputs.shape[0]):
            correct = np.argmax(outputs[number]) == label[number]
            total_correct += correct
        print('Accuracy: ' + str(total_correct/outputs.shape[0] * 100) + '%')
accuracy_calc(X_train_torch, y_train_torch)
accuracy_calc(X_test_torch, y_test_torch)