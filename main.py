import yaml
import copy
import logging
import argparse
import numpy as np
from sklearn.metrics import classification_report

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
    elif isinstance(train_test_split, dict):
        train_indices, test_indices = train_test_set.sample(train_test_split)

        train_set = torch.utils.data.Subset(train_test_set, train_indices)
        test_set = torch.utils.data.Subset(train_test_set, test_indices)

    test_set.__getattribute__('dataset').__setattr__('transform', test_transform)

    print(f'train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}')

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=32)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True, num_workers=32)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=32)

    return (train_loader, val_loader, test_loader)


def load_net(output_classes, model_name, net=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name == 'resnet':
        if net == None:
            net = models.resnet18(pretrained=True)
            net.fc = nn.Sequential(
                nn.Linear(net.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, output_classes)
            )
        else:
            net.fc[-1] == nn.Linear(net.fc[-1].in_features, output_classes)
    elif model_name == 'efficientnet':
        if net == None:
            net = EfficientNet.from_pretrained('efficientnet-b0')
            net._fc = nn.Linear(net._fc.in_features, output_classes)
        else:
            net._fc = nn.Linear(net._fc.in_features, output_classes)

    if net == None:
        for param in list(net.parameters())[: -15]:
            param.requires_grad = False
    else:
        for param in net.parameters():
            param.requires_grad = True

    net = net.to(device)

    return net


def run(net, category_filter, train_test_split, patience, moniter, is_query=False):
    output_classes = len(category_filter)
    train_loader, val_loader, test_loader = load_data(image_size, category_filter, train_test_split)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, moniter=moniter)

    model = Model(net, optimizer, criterion)
    # model.summary((1, 3) + image_size)

    model.train(train_loader, epochs=30, val_loader=val_loader, scheduler=scheduler, early_stopping=early_stopping)

    if is_query == False:
        history = model.test(test_loader)

        return (model, history)
    else:
        x_test, y_test = [], []
        labels = test_loader.dataset.dataset.labels()

        for image, label in test_loader.dataset:
            x_test.append(image.numpy())
            y_test.append(label)

        y_hat = model.predict_class(x_test)

        for i in range(len(y_test)):
            y_test[i] = labels[y_test[i]]
            y_hat[i] = labels[y_hat[i]]

        return (y_test, y_hat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, default='resnet')
    parser.add_argument('-c', '--config', dest='config_name', type=str, default='config.yaml')
    args = parser.parse_args()

    logger = logging.getLogger('Processing Logger')
    logger.setLevel(logging.DEBUG)

    log_stream = logging.StreamHandler()
    log_stream.setLevel(logging.DEBUG)

    log_file = logging.FileHandler('process.log')
    log_file.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_stream.setFormatter(formatter)
    log_file.setFormatter(formatter)

    logger.addHandler(log_stream)
    logger.addHandler(log_file)

    logger.info('-' * 100)
    logger.info(args.config_name)

    image_size = (240, 240)

    with open(args.config_name, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    net = load_net(len(config['train']), args.model_name)

    model, history = run(net, config['train'], train_test_split=0.75, patience=7, moniter='val_loss')
    model.export(f'backup/model_{args.config_name.split(".")[0]}.onnx', ((1, 3) + image_size))

    logger.info(f'train accuracy: {(history["accuracy"]):.3f}')

    total = 0
    correct = 0

    for category_filter in config['test']:
        net_copy = load_net(len(category_filter['class']), args.model_name, net=copy.deepcopy(net))

        train_test_split = {}
        for i in range(len(category_filter['class'])):
            train_test_split[category_filter['class'][i]] = category_filter['sample'][i]

        logger.info(f'train test split: {train_test_split}')

        y_test, y_hat = run(net_copy, category_filter['class'], train_test_split=train_test_split, patience=4, moniter='loss', is_query=True)

        total += len(y_test)
        correct += (y_test == y_hat).sum()

        logger.info(f'\n{classification_report(y_test, y_hat)}')

    logger.info(f'average accuracy: {(correct / total):.3f}')

    logger.info('-' * 100)
