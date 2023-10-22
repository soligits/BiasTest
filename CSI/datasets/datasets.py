import os

import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from utils.utils import set_random_seed

from glob import glob
from torch.utils.data import Dataset, ConcatDataset
import os
import random
from PIL import Image

class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, train=True, count=-1):
        self.transform = transform
        self.image_files = []
        if train:
            self.image_files = glob(os.path.join(root, category, "train", "good", "*.png"))
        else:
            image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
            normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
            anomaly_image_files = list(set(image_files) - set(normal_image_files))
            self.image_files = image_files
        if count != -1:
            if count<len(self.image_files):
                self.image_files = self.image_files[:count]
            else:
                t = len(self.image_files)
                for i in range(count-t):
                    self.image_files.append(random.choice(self.image_files[:t]))
        self.image_files.sort(key=lambda y: y.lower())
        self.train = train
        self.targets = []
        for image_file in self.image_files:
            if os.path.dirname(image_file).endswith("good"):
                target = 0
            else:
                target = 1
            self.targets.append(target)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        target = self.targets[index]
        return image, target

    def __len__(self):
        return len(self.image_files)

DATA_PATH = "~/data/"
IMAGENET_PATH = "~/data/ImageNet"


CIFAR10_SUPERCLASS = list(range(10))  # one class
SVHN_SUPERCLASS = list(range(10))
MNIST_SUPERCLASS = list(range(10))
FASHION_SUPERCLASS = list(range(10))
MVTEC_SUPERCLASS = list(range(15))
IMAGENET_SUPERCLASS = list(range(30))  # one class

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]


class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform1 = transform
        self.transform2 = transform

    def __call__(self, sample):
        x1 = self.transform1(sample)
        x2 = self.transform2(sample)
        return x1, x2


class MultiDataTransformList(object):
    def __init__(self, transform, clean_trasform, sample_num):
        self.transform = transform
        self.clean_transform = clean_trasform
        self.sample_num = sample_num

    def __call__(self, sample):
        set_random_seed(0)

        sample_list = []
        for i in range(self.sample_num):
            sample_list.append(self.transform(sample))

        return sample_list, self.clean_transform(sample)


def get_transform(image_size=None, bw=False):
    # Note: data augmentation is implemented in the layers
    # Hence, we only define the identity transformation here

    if image_size:  # use pre-specified image size
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size[0], image_size[1])),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
            ]
        )
    else:  # use default image size
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.ToTensor()

    return train_transform, test_transform


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


def get_transform_imagenet():
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    train_transform = MultiDataTransform(train_transform)

    return train_transform, test_transform


train_transform_bw = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
test_transform_bw = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)


def get_dataset(
    P, dataset, test_only=False, image_size=None, download=True, eval=False
):
    if dataset in [
        "imagenet",
        "cub",
        "stanford_dogs",
        "flowers102",
        "places365",
        "food_101",
        "caltech_256",
        "dtd",
        "pets"
    ]:
        if eval:
            train_transform, test_transform = get_simclr_eval_transform_imagenet(
                P.ood_samples, P.resize_factor, P.resize_fix
            )
        else:
            train_transform, test_transform = get_transform_imagenet()
    elif dataset == 'mvtec':
        train_transform, test_transform = get_transform_imagenet()
    else:
        train_transform, test_transform = get_transform(image_size=image_size)


    if dataset == "cifar10":
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.CIFAR10(
            DATA_PATH, train=True, download=download, transform=train_transform
        )
        test_set = datasets.CIFAR10(
            DATA_PATH, train=False, download=download, transform=test_transform
        )

    elif dataset == "mnist":
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.MNIST(
            DATA_PATH, train=True, download=download, transform=train_transform_bw
        )
        test_set = datasets.MNIST(
            DATA_PATH, train=False, download=download, transform=test_transform_bw
        )

    elif dataset == "fashion":
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.FashionMNIST(
            DATA_PATH, train=True, download=download, transform=train_transform_bw
        )
        test_set = datasets.FashionMNIST(
            DATA_PATH, train=False, download=download, transform=test_transform_bw
        )

    elif dataset == "cifar100":
        image_size = (32, 32, 3)
        n_classes = 100
        train_set = datasets.CIFAR100(
            DATA_PATH, train=True, download=download, transform=train_transform
        )
        test_set = datasets.CIFAR100(
            DATA_PATH, train=False, download=download, transform=test_transform
        )

    elif dataset == "svhn":
        image_size = (32, 32, 3)
        n_classes = 10
        train_set = datasets.SVHN(
            DATA_PATH, split="train", download=download, transform=test_transform
        )
        train_set.targets = train_set.labels
        test_set = datasets.SVHN(
            DATA_PATH, split="test", download=download, transform=test_transform
        )
        test_set.targets = test_set.labels
    
    elif dataset == "mvtec":
        image_size = (224, 224, 3)
        n_classes = 15
        train_dataset = []
        test_dataset = []
        train_targets = []
        test_targets = []
        CLASS_NAMES = ['toothbrush', 'zipper', 'transistor', 'tile', 'grid', 'wood', 'pill', 'bottle', 'capsule', 'metal_nut', 'hazelnut', 'screw', 'carpet', 'leather', 'cable']
        for i, cat in enumerate(CLASS_NAMES):
            if i in P.one_class_idx:
                tr = MVTecDataset(root='data/mvtec_anomaly_detection', train=True, category=cat, transform=train_transform, count=-1)
                te = MVTecDataset(root='data/mvtec_anomaly_detection', train=False, category=cat, transform=test_transform, count=-1)
                train_dataset.append(tr)
                test_dataset.append(te)
                train_targets += tr.targets
                test_targets += te.targets

        train_set = ConcatDataset(train_dataset)
        train_set.targets = train_targets
        test_set = ConcatDataset(test_dataset)
        test_set.targets = test_targets
    

    elif dataset == 'dtd':
        image_size = (32, 32, 3)
        n_classes = 47
        test_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        train_transform = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
        ])
        train_set = datasets.DTD('./data', split="train", download=True, transform=train_transform)
        test_set = datasets.DTD('./data', split="test", download=True, transform=test_transform)
        train_set.targets = train_set._labels
        test_set.targets = test_set._labels
        
    elif dataset == "lsun_resize":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "LSUN_resize")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == "lsun_fix":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "LSUN_fix")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == "imagenet_resize":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "Imagenet_resize")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == "imagenet_fix":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "Imagenet_fix")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == "imagenet":
        image_size = (224, 224, 3)
        n_classes = 30
        train_dir = os.path.join(IMAGENET_PATH, "one_class_train")
        test_dir = os.path.join(IMAGENET_PATH, "one_class_test")
        train_set = datasets.ImageFolder(train_dir, transform=train_transform)
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)

    elif dataset == "stanford_dogs":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "stanford_dogs")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == "cub":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "cub200")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == "flowers102":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "flowers102")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == "places365":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "places365")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == "food_101":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "food-101", "images")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == "caltech_256":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "caltech-256")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == "dtd":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "dtd", "images")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    elif dataset == "pets":
        assert test_only and image_size is not None
        test_dir = os.path.join(DATA_PATH, "pets")
        test_set = datasets.ImageFolder(test_dir, transform=test_transform)
        test_set = get_subset_with_len(test_set, length=3000, shuffle=True)

    else:
        raise NotImplementedError()

    if test_only:
        return test_set
    else:
        return train_set, test_set, image_size, n_classes


def get_superclass_list(dataset):
    if dataset == "cifar10":
        return CIFAR10_SUPERCLASS
    elif dataset == "cifar100":
        return CIFAR100_SUPERCLASS
    elif dataset == "imagenet":
        return IMAGENET_SUPERCLASS
    elif dataset == 'fashion':
        return FASHION_SUPERCLASS
    elif dataset == 'mnist':
        return MNIST_SUPERCLASS
    elif dataset == 'mvtec':
        return MVTEC_SUPERCLASS
    elif dataset == 'svhn':
        return SVHN_SUPERCLASS
    
    else:
        raise NotImplementedError()


def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def get_simclr_eval_transform_imagenet(sample_num, resize_factor, resize_fix):
    resize_scale = (resize_factor, 1.0)  # resize scaling factor
    if resize_fix:  # if resize_fix is True, use same scale
        resize_scale = (resize_factor, resize_factor)

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=resize_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    clean_trasform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    transform = MultiDataTransformList(transform, clean_trasform, sample_num)

    return transform, transform
