import yaml
import copy
import argparse
import numpy as np

import torch
import torch.nn as nn
from torchvision import models, transforms

from Model import Model, EarlyStopping
from autoaugment import ImageNetPolicy
from efficientnet_pytorch import EfficientNet
from InaturalistDataset import InaturalistDataset


def load_data(image_size, category_filter, train_test_split):
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(30, translate=(0.3, 0.3), scale=(1.0, 1.5)),
        ImageNetPolicy(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    root_dir = '/train-data/inaturalist-2019/'
    train_test_set = InaturalistDataset(
        root_dir + 'train2019.json',
        root_dir,
        train_transform,
        category_filter=category_filter
    )
    val_set = InaturalistDataset(
        root_dir + 'val2019.json',
        root_dir,
        test_transform,
        category_filter=category_filter
    )

    if isinstance(train_test_split, float):
        train_size = int(len(train_test_set) * train_test_split)
        test_size = len(train_test_set) - train_size

        train_set, test_set = torch.utils.data.random_split(train_test_set, [train_size, test_size])
    elif isinstance(train_test_split, int):
        test_indices = train_test_set.sample(train_test_split)

        train_indices = list(set(range(len(train_test_set))) - set(test_indices))
        train_indices = np.random.permutation(train_indices)

        train_set = torch.utils.data.Subset(train_test_set, train_indices[: 500])
        test_set = torch.utils.data.Subset(train_test_set, test_indices)

    test_set.__getattribute__('dataset').__setattr__('transform', test_transform)

    print(f'train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=32)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True, num_workers=32)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=32)

    return (train_loader, val_loader, test_loader)


def load_net(output_classes, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'resnet':
        net = models.resnet18(pretrained=True)
        net.fc = nn.Sequential(
            nn.Linear(net.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_classes)
        )
    elif model_name == 'efficientnet':
        net = EfficientNet.from_pretrained('efficientnet-b0')
        net._fc = nn.Linear(net._fc.in_features, output_classes)

    for param in list(net.parameters())[: -15]:
        param.requires_grad = False

    net = net.to(device)

    return net


def update_net(net, output_classes, model_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'resnet':
        net.fc[-1] = nn.Linear(net.fc[-1].in_features, output_classes)
    elif model_name == 'efficientnet':
        net._fc = nn.Linear(net._fc.in_features, output_classes)

    for param in net.parameters():
        param.requires_grad = True

    net = net.to(device)

    return net


def run(net, category_filter, train_test_split, patience, moniter):
    output_classes = len(category_filter)
    train_loader, val_loader, test_loader = load_data(image_size, category_filter, train_test_split)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, moniter=moniter)

    model = Model(net, optimizer, criterion)
    # model.summary((1, 3) + image_size)

    train_history = model.train(train_loader, 30, val_loader=val_loader, scheduler=scheduler, early_stopping=early_stopping)
    test_history = model.test(test_loader)

    return (train_history, test_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, default='resnet')
    args = parser.parse_args()

    image_size = (240, 240)
    model_name = args.model_name

    with open('config.yaml', 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    net = load_net(len(config['allow']['train']), model_name)

    train_history, test_history = run(net, config['allow']['train'], train_test_split=0.75, patience=7, moniter='val_loss')

    loss_result = 0
    accuracy_result = 0
    for category_filter in config['allow']['test']:
        net_copy = update_net(copy.deepcopy(net), len(category_filter), model_name)

        train_history, test_history = run(net_copy, category_filter, train_test_split=10, patience=4, moniter='loss')

        loss_result += test_history['loss']
        accuracy_result += test_history['accuracy']

    print(f'average loss: {loss_result / len(config["allow"]["test"])}, average accuracy: {accuracy_result / len(config["allow"]["test"])}')
