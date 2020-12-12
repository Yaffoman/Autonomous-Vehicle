import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import model
import datasets
import matplotlib.pyplot as plt
from torch.utils.data.dataset import random_split
import torchvision.models as models
from tqdm.notebook import tqdm

# Make sure to use the GPU. The following line is just a check to see if GPU is availables
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# load your dataset and dataloader
# feel free to change header of bird_dataset class
root = 'birds_dataset/'
train_dataset = datasets.ArrowDataset(device)
val_dataset = datasets.ArrowDataset(device, val=50)
# test_dataset = datasets.ArrowDataset(device, test=50)
test_dataset = datasets.TestingDataset(device)

# Fill in optional arguments to the dataloader as you need it
train_dataloader = DataLoader(train_dataset, batch_size=300, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)

# Create NN model object
nn_model = model.baseline_Net(classes=2)
nn_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-5, weight_decay=.2)


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


nn_model = nn_model.apply(weights_init)

# train your model
# For each epoch iterate over your dataloaders/datasets, pass it to your NN model, get output, calculate loss and
# backpropagate using optimizer
epoch_acc_1 = []
epoch_loss_1 = []
val_acc_1 = []
val_loss_1 = []
epoch_acc_2 = []
epoch_loss_2 = []
val_acc_2 = []
val_loss_2 = []
best_val_acc = 0.0
PATH = 'best_model'

num_epochs = 50

# use first val set
for epoch in range(num_epochs):
    batch_acc = []
    batch_loss = []
    for x, y in train_dataloader:
        optimizer.zero_grad()  # zero the gradient buffers
        nn_model.zero_grad()
        out = nn_model.forward(x)
        preds = torch.argmax(out, axis=1)

        acc = nn_model.accuracy(preds, y)
        batch_acc.append(acc)

        loss = criterion(out, y)
        batch_loss.append(loss.item())

        loss.backward()
        optimizer.step()

    epoch_loss_1.append(sum(batch_loss) / len(batch_loss))
    epoch_acc_1.append(sum(batch_acc) / len(batch_acc))

    batch_acc = []
    batch_loss = []
    for x, y in val_dataloader:
        optimizer.zero_grad()  # zero the gradient buffers
        nn_model.zero_grad()
        out = nn_model.forward(x)
        preds = torch.argmax(out, axis=1)

        acc = nn_model.accuracy(preds, y)
        batch_acc.append(acc)

        loss = criterion(out, y)
        batch_loss.append(loss.item())

    val_loss_1.append(sum(batch_loss) / len(batch_loss))
    val_acc_1.append(sum(batch_acc) / len(batch_acc))

    print(epoch, 'acc', epoch_acc_1[epoch], val_acc_1[epoch])
    print('  ', 'loss', epoch_loss_1[epoch], val_loss_1[epoch])

figure, axes = plt.subplots()
axes.set_xlabel('Epochs')
axes.set_ylabel('Accuracy')
axes.set_title('Training and Validation Accuracy')
axes.plot(epoch_acc_1, label = 'Training Acc')
axes.plot(val_acc_1, label = 'Validation Acc')

axes.set_xlabel('Epochs')
axes.set_ylabel('Loss')
axes.set_title('Training and Validation Loss')
axes.plot(epoch_loss_1, label = 'Training Loss')
axes.plot(val_loss_1, label = 'Validation Loss')

axes.legend()
plt.show()
plt.close(figure)



batch_acc = []
for x,y in test_dataloader:  # Check Test acc
    out = nn_model.forward(x)
    preds = torch.argmax(out, axis=1)
    print(preds)
    acc = nn_model.accuracy(preds,y)
    batch_acc.append(acc)

print("Test Accuracy: ", sum(batch_acc)/len(batch_acc))
