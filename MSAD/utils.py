import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
import torchvision.models as models
import torch.nn.functional as F
from PIL import ImageFilter
import random
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST, CIFAR100
import os
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from torchvision import models
import requests
import subprocess


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


transform_color = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
)

transform_resnet18_color = transforms.Compose(
    [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


transform_bw = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)

transform_resnet18_bw = transforms.Compose(
    [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)


moco_transform_color = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

moco_transform_bw = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

mu = torch.tensor(mean).view(3, 1, 1).cuda()
std = torch.tensor(std).view(3, 1, 1).cuda()


class Transform:
    def __init__(self, bw=False):
        self.moco_transform = moco_transform_bw if bw else moco_transform

    def __call__(self, x):
        x_1 = self.moco_transform(x)
        x_2 = self.moco_transform(x)
        return x_1, x_2


class Model(torch.nn.Module):
    def __init__(self, backbone="18"):
        super().__init__()
        self.norm = lambda x: (x - mu) / std
        if backbone == "152":
            self.backbone = models.resnet152(pretrained=True)
        elif backbone == "50":
            self.backbone = models.resnet50(pretrained=True)
        elif backbone == "18":
            self.backbone = models.resnet18(pretrained=True)
        else:
            raise Exception("Source Dataset is not supported yet. ")

        self.backbone.fc = torch.nn.Identity()
        freeze_parameters(self.backbone, backbone, train_fc=False)

    def forward(self, x):
        x = self.norm(x)
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n


def freeze_parameters(model, backbone, train_fc=False):
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False
    if backbone == 152:
        for p in model.conv1.parameters():
            p.requires_grad = False
        for p in model.bn1.parameters():
            p.requires_grad = False
        for p in model.layer1.parameters():
            p.requires_grad = False
        for p in model.layer2.parameters():
            p.requires_grad = False


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_loaders(dataset, label_classes, batch_size, dataset_path, backbone):
    trainset, trainset_moco = get_train_dataset(
        dataset, label_classes, dataset_path, backbone
    )

    print(
        f"Train Dataset: {dataset}, Normal Classes: {label_classes}, length Trainset: {len(trainset)}"
    )

    testset = get_test_dataset(dataset, label_classes, dataset_path, backbone)

    print(f"length Testset: {len(testset)}")

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    train_loader_moco = torch.utils.data.DataLoader(
        trainset_moco, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, train_loader_moco


def get_train_dataset(dataset, label_class, path, backbone):
    if dataset == "cifar10":
        return get_CIFAR10_train(label_class, path, backbone)
    elif dataset == "cifar100":
        return get_CIFAR100_train(label_class, path, backbone)
    elif dataset == "mnist":
        return get_MNIST_train(label_class, path, backbone)
    elif dataset == "fashion":
        return get_FASHION_MNIST_train(label_class, path, backbone)
    elif dataset == "svhn":
        return get_SVHN_train(label_class, path, backbone)
    else:
        raise Exception("Source Dataset is not supported yet. ")
        exit()


def get_test_dataset(dataset, normal_labels, path, backbone):
    if dataset == "cifar10":
        return get_CIFAR10_test(normal_labels, path, backbone)
    elif dataset == "cifar100":
        return get_CIFAR100_test(normal_labels, path, backbone)
    elif dataset == "mnist":
        return get_MNIST_test(normal_labels, path, backbone)
    elif dataset == "fashion":
        return get_FASHION_MNIST_test(normal_labels, path, backbone)
    elif dataset == "svhn":
        return get_SVHN_test(normal_labels, path, backbone)
    else:
        raise Exception("Target Dataset is not supported yet. ")
        exit()


def get_CIFAR10_train(normal_class_labels, path, backbone):
    transform = transform_color if backbone == "152" else transform_resnet18_color

    trainset = CIFAR10(root=path, train=True, download=True, transform=transform)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    trainset_moco = CIFAR10(root=path, train=True, download=True, transform=Transform())

    normal_mask = np.isin(trainset_moco.targets, normal_class_labels)

    trainset_moco.data = trainset_moco.data[normal_mask]
    trainset_moco.targets = [0 for _ in trainset_moco.targets]

    return trainset, trainset_moco


def get_CIFAR10_test(normal_class_labels, path, backbone):
    transform = transform_color if backbone == "152" else transform_resnet18_color

    testset = CIFAR10(root=path, train=False, download=True, transform=transform)
    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array(
        [
            4,
            1,
            14,
            8,
            0,
            6,
            7,
            7,
            18,
            3,
            3,
            14,
            9,
            18,
            7,
            11,
            3,
            9,
            7,
            11,
            6,
            11,
            5,
            10,
            7,
            6,
            13,
            15,
            3,
            15,
            0,
            11,
            1,
            10,
            12,
            14,
            16,
            9,
            11,
            5,
            5,
            19,
            8,
            8,
            15,
            13,
            14,
            17,
            18,
            10,
            16,
            4,
            17,
            4,
            2,
            0,
            17,
            4,
            18,
            17,
            10,
            3,
            2,
            12,
            12,
            16,
            12,
            1,
            9,
            19,
            2,
            10,
            0,
            1,
            16,
            12,
            9,
            13,
            15,
            13,
            16,
            19,
            2,
            4,
            6,
            19,
            5,
            5,
            8,
            19,
            18,
            1,
            2,
            15,
            6,
            0,
            17,
            8,
            14,
            13,
        ]
    )
    return coarse_labels[targets]


def get_CIFAR100_train(normal_class_labels, path, backbone):
    transform = transform_color if backbone == "152" else transform_resnet18_color

    trainset = CIFAR100(root=path, train=True, download=True, transform=transform)
    trainset.targets = sparse2coarse(trainset.targets)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    trainset_moco = CIFAR100(
        root=path, train=True, download=True, transform=Transform()
    )
    trainset_moco.targets = sparse2coarse(trainset_moco.targets)

    normal_mask = np.isin(trainset_moco.targets, normal_class_labels)

    trainset_moco.data = trainset_moco.data[normal_mask]
    trainset_moco.targets = [0 for _ in trainset_moco.targets]

    return trainset, trainset_moco


def get_CIFAR100_test(normal_class_labels, path, backbone):
    transform = transform_color if backbone == "152" else transform_resnet18_color

    testset = CIFAR100(root=path, train=False, download=True, transform=transform)
    testset.targets = sparse2coarse(testset.targets)

    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_MNIST_train(normal_class_labels, path, backbone):
    transform = transform_bw if backbone == "152" else transform_resnet18_bw

    trainset = MNIST(root=path, train=True, download=True, transform=transform)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    trainset_moco = MNIST(
        root=path, train=True, download=True, transform=Transform(bw=True)
    )

    normal_mask = np.isin(trainset_moco.targets, normal_class_labels)

    trainset_moco.data = trainset_moco.data[normal_mask]
    trainset_moco.targets = [0 for _ in trainset_moco.targets]

    return trainset, trainset_moco


def get_MNIST_test(normal_class_labels, path, backbone):
    transform = transform_bw if backbone == "152" else transform_resnet18_bw

    testset = MNIST(root=path, train=False, download=True, transform=transform)
    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_FASHION_MNIST_train(normal_class_labels, path, backbone):
    transform = transform_bw if backbone == "152" else transform_resnet18_bw

    trainset = FashionMNIST(root=path, train=True, download=True, transform=transform)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    trainset_moco = FashionMNIST(
        root=path, train=True, download=True, transform=Transform(bw=True)
    )

    normal_mask = np.isin(trainset_moco.targets, normal_class_labels)

    trainset_moco.data = trainset_moco.data[normal_mask]
    trainset_moco.targets = [0 for _ in trainset_moco.targets]

    return trainset, trainset_moco


def get_FASHION_MNIST_test(normal_class_labels, path, backbone):
    transform = transform_bw if backbone == "152" else transform_resnet18_bw

    testset = FashionMNIST(root=path, train=False, download=True, transform=transform)

    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_SVHN_train(normal_class_labels, path, backbone):
    transform = transform_color if backbone == "152" else transform_resnet18_color

    trainset = SVHN(root=path, split="train", download=True, transform=transform)

    normal_mask = np.isin(trainset.labels, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.labels = [0 for _ in trainset.labels]

    trainset_moco = SVHN(root=path, split="train", download=True, transform=Transform())

    normal_mask = np.isin(trainset_moco.labels, normal_class_labels)

    trainset_moco.data = trainset_moco.data[normal_mask]
    trainset_moco.labels = [0 for _ in trainset_moco.labels]

    return trainset, trainset_moco


def get_SVHN_test(normal_class_labels, path, backbone):
    transform = transform_color if backbone == "152" else transform_resnet18_color

    testset = SVHN(root=path, split="test", download=True, transform=transform)
    test_mask = np.isin(testset.labels, normal_class_labels)

    testset.labels = np.array(testset.labels)
    testset.labels[test_mask] = 0
    testset.labels[~test_mask] = 1

    return testset
