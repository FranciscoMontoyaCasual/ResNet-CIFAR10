import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets
from torchvision import transforms as T

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

train_cifar = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

test_cifar = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)

train_dataloader = DataLoader(train_cifar, batch_size=200, shuffle=True)
test_dataloader = DataLoader(test_cifar, batch_size=1, shuffle=True)

model = models.resnet50(pretrained=True).to(device)
new_model = nn.Sequential(*list(model.children())[:-1])

for i, parameter in enumerate(new_model.parameters()):
    parameter.requires_grad = False

new_model = nn.Sequential(
    *list(new_model.children()),
    nn.Flatten(),
    nn.Linear(in_features=2048, out_features=256, bias=True),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=10)
).to(device)

def get_accuracy(model, dataloader, device):
    correct_pred = 0
    n = len(dataloader.dataset)

    with torch.no_grad():
        model.eval()

        for x, y_true in dataloader:
            x = x.to(device)
            y_true = y_true.to(device)

            y_hat = model(x)
            y_prob = nn.Softmax(dim=1)(y_hat)
            _, pred_labels = torch.max(y_prob, 1)

            correct_pred += (pred_labels == y_true).sum()

    return float(correct_pred) / n

def train(train_loader, model, criterion, optimiser, device):
    model.train()
    running_loss = 0

    for x, y_true in train_loader:
        x = x.to(device)
        y_true = y_true.to(device)

        y_hat = model(x)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * x.size(0)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimiser, epoch_loss

def validate(test_loader, model, criterion, device):
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for x, y_true in test_loader:
            x = x.to(device)
            y_true = y_true.to(device)

            y_hat = model(x)
            loss = criterion(y_hat, y_true)
            running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)
    return model, epoch_loss

def train_loop(model, criterion, optimiser, train_loader, test_loader, epochs, device):
    for epoch in range(epochs):
        model, optimiser, train_loss = train(train_loader, model, criterion, optimiser, device)

        with torch.no_grad():
            model, test_loss = validate(test_loader, model, criterion, device)

            train_acc = get_accuracy(model, train_loader, device)
            test_acc = get_accuracy(model, test_loader, device)

            print(f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {test_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * test_acc:.2f}')
            
    return model, optimiser

optimiser = torch.optim.Adam(new_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

new_model, optimiser = train_loop(new_model, criterion, optimiser, train_dataloader, test_dataloader, 10, device)