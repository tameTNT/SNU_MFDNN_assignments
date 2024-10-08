import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.datasets import MNIST

from torchvision.transforms import transforms as T
import time

###################
# Hyperparameters #
###################
learning_rate = 0.1
batch_size = 64
epochs = 150

#############################################
# Fix torch random seed for stability issue #
#############################################

torch.manual_seed(0)

#################################
# STEP 1: Prepare MNIST dataset #
#################################

try:
    dataset = MNIST('./data', train=True, transform=T.ToTensor())
except:
    dataset = MNIST('./data', train=True, download=True, transform=T.ToTensor())

# Randomly Select 1/10 Samples
total_data_num = len(dataset)
sub_data_ind = torch.randperm(total_data_num)
sub_data_ind = sub_data_ind[:total_data_num // 10]
dataset.data = dataset.data[sub_data_ind]

# Create Random Label
random_label = torch.randint(0, 10, size=(len(dataset.data),))
dataset.targets = random_label

# Prepare DataLoader
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


#######################################################
# STEP 2 : Define Model (Modified version of AlexNet) #
#######################################################

class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc_layer1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6400, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, 10)
        )

    def forward(self, x):
        output = self.conv_layer1(x)
        output = self.conv_layer2(output)
        output = self.conv_layer3(output)
        output = torch.flatten(output, 1)
        output = self.fc_layer1(output)
        return output


###########################
# Step 3: Train the Model #
###########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# Function to check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            scores = model(images)
            preds = torch.argmax(scores, 1)

            num_correct += (preds == labels).sum().item()
            num_samples += preds.size(0)
        accuracy = float(num_correct) / num_samples
        print(f'Accuracy: {num_correct} / {num_samples} ({100 * accuracy:.2f}%)')
        return accuracy


model.train()
train_loss = []
train_acc = []

tick = time.time()
for epoch in range(150):
    print(f"\nEpoch {epoch + 1} / {epochs}")
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()

        optimizer.step()

    print(f"Epoch {epoch + 1} / {epochs} Training Loss: {loss.item():.4f}")
    train_loss.append(loss.item())
    train_acc.append(check_accuracy(train_loader, model))

tock = time.time()
print(f"Total Time Spent for Training: {tock - tick}")

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.plot(train_acc, label="Train Accuracy", color=color)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy", color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.plot(train_loss, label="Train Loss", color=color)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss", color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training with Randomized Label')
fig.legend()
plt.show()