import torch
import torchvision
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST, CIFAR100
import torchvision.transforms as transforms
import numpy as np
import faiss
import ResNet

transform_color = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_gray = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_resnet_model(resnet_type=152):
    """
    A function that returns the required pre-trained resnet model
    :param resnet_number: the resnet type
    :return: the pre-trained model
    """
    if resnet_type == 18:
        return ResNet.resnet18(pretrained=True, progress=True)
    elif resnet_type == 50:
        return ResNet.wide_resnet50_2(pretrained=True, progress=True)
    elif resnet_type == 101:
        return ResNet.resnet101(pretrained=True, progress=True)
    else:  # 152
        return ResNet.resnet152(pretrained=True, progress=True)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def freeze_parameters(model, train_fc=False):
    for p in model.conv1.parameters():
        p.requires_grad = False
    for p in model.bn1.parameters():
        p.requires_grad = False
    for p in model.layer1.parameters():
        p.requires_grad = False
    for p in model.layer2.parameters():
        p.requires_grad = False
    if not train_fc:
        for p in model.fc.parameters():
            p.requires_grad = False


def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)


def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(
        root="./data/tiny", transform=transform_color
    )
    outlier_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return outlier_loader


def get_loaders(dataset, label_classes, batch_size, dataset_path):
    trainset = get_train_dataset(dataset, label_classes, dataset_path)

    print(
        f"Train Dataset: {dataset}, Normal Classes: {label_classes}, length Trainset: {len(trainset)}"
    )

    testset = get_test_dataset(dataset, label_classes, dataset_path)

    print(f"length Testset: {len(testset)}")

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def get_train_dataset(dataset, label_class, path):
    if dataset == "cifar10":
        return get_CIFAR10_train(label_class, path)
    elif dataset == "cifar100":
        return get_CIFAR100_train(label_class, path)
    elif dataset == "mnist":
        return get_MNIST_train(label_class, path)
    elif dataset == "fashion":
        return get_FASHION_MNIST_train(label_class, path)
    elif dataset == "svhn":
        return get_SVHN_train(label_class, path)
    else:
        raise Exception("Source Dataset is not supported yet. ")
        exit()


def get_test_dataset(dataset, normal_labels, path):
    if dataset == "cifar10":
        return get_CIFAR10_test(normal_labels, path)
    elif dataset == "cifar100":
        return get_CIFAR100_test(normal_labels, path)
    elif dataset == "mnist":
        return get_MNIST_test(normal_labels, path)
    elif dataset == "fashion":
        return get_FASHION_MNIST_test(normal_labels, path)
    elif dataset == "svhn":
        return get_SVHN_test(normal_labels, path)
    else:
        raise Exception("Target Dataset is not supported yet. ")
        exit()


def get_CIFAR10_train(normal_class_labels, path):
    trainset = CIFAR10(root=path, train=True, download=True, transform=transform_color)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    return trainset


def get_CIFAR10_test(normal_class_labels, path):
    testset = CIFAR10(root=path, train=False, download=True, transform=transform_color)
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


def get_CIFAR100_train(normal_class_labels, path):
    trainset = CIFAR100(root=path, train=True, download=True, transform=transform_color)
    trainset.targets = sparse2coarse(trainset.targets)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    return trainset


def get_CIFAR100_test(normal_class_labels, path):
    testset = CIFAR100(root=path, train=False, download=True, transform=transform_color)
    testset.targets = sparse2coarse(testset.targets)

    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_MNIST_train(normal_class_labels, path):
    trainset = MNIST(root=path, train=True, download=True, transform=transform_gray)

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    return trainset


def get_MNIST_test(normal_class_labels, path):
    testset = MNIST(root=path, train=False, download=True, transform=transform_gray)
    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_FASHION_MNIST_train(normal_class_labels, path):
    trainset = FashionMNIST(
        root=path, train=True, download=True, transform=transform_gray
    )

    normal_mask = np.isin(trainset.targets, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.targets = [0 for _ in trainset.targets]

    return trainset


def get_FASHION_MNIST_test(normal_class_labels, path):
    testset = FashionMNIST(
        root=path, train=False, download=True, transform=transform_gray
    )

    test_mask = np.isin(testset.targets, normal_class_labels)

    testset.targets = np.array(testset.targets)
    testset.targets[test_mask] = 0
    testset.targets[~test_mask] = 1

    return testset


def get_SVHN_train(normal_class_labels, path):
    trainset = SVHN(root=path, split="train", download=True, transform=transform_color)

    normal_mask = np.isin(trainset.labels, normal_class_labels)

    trainset.data = trainset.data[normal_mask]
    trainset.labels = [0 for _ in trainset.labels]

    return trainset


def get_SVHN_test(normal_class_labels, path):
    testset = SVHN(root=path, split="test", download=True, transform=transform_color)
    test_mask = np.isin(testset.labels, normal_class_labels)

    testset.labels = np.array(testset.labels)
    testset.labels[test_mask] = 0
    testset.labels[~test_mask] = 1

    return testset


def clip_gradient(optimizer, grad_clip):
    assert grad_clip > 0, "gradient clip value must be greater than 1"
    for group in optimizer.param_groups:
        for param in group["params"]:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)
