import torch
import torch.nn as nn
from torchvision import models, transforms

from Model import Model, EarlyStopping
from InaturalistDataset import InaturalistDataset


def load_data(image_size):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(0, translate=(0.3, 0.3), scale=(1.0, 1.5)),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    root_dir = '/train-data/inaturalist-2019/'
    train_val_set = InaturalistDataset(
        root_dir + 'train2019.json',
        root_dir,
        train_transform,
        # category_filter='Birds'
    )
    test_set = InaturalistDataset(
        root_dir + 'val2019.json',
        root_dir,
        test_transform,
        # category_filter='Birds'
    )

    train_size = int(len(train_val_set) * 0.75)
    val_size = len(train_val_set) - train_size

    train_set, val_set = torch.utils.data.random_split(train_val_set, [train_size, val_size])
    val_set.__getattribute__('dataset').__setattr__('transform', test_transform)

    print(f'train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=32)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True, num_workers=32)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=32)

    return (train_loader, val_loader, test_loader)


def load_net():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = models.resnet18(pretrained=True)

    # for i, child in enumerate(net.children()):
    #     if i < 8:
    #         for param in child.parameters():
    #             param.requires_grad = False

    for i, param in enumerate(net.parameters()):
        if i < 55:
            param.requires_grad = False

    net.fc = nn.Sequential(
        nn.Linear(net.fc.in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1010)
    )

    net = net.to(device)

    return net


if __name__ == '__main__':
    image_size = 80

    train_loader, val_loader, test_loader = load_data(image_size)

    net = load_net()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=7, moniter='val_loss')

    model = Model(net, optimizer, criterion)
    model.summary((1, 3, image_size, image_size))

    model.train(train_loader, 50, val_loader=val_loader, early_stopping=early_stopping)
    model.test(test_loader)
