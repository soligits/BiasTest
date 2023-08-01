import torch

from util.datasets import get_datasets, CIFAR100OE


def load_dataset(config):
    trainset, testset = get_datasets(
        config.dataset, config.normal_classes, config.data_path
    )

    cifar100 = CIFAR100OE(root=config.data_path)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=config.batch_size // 2,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=100, num_workers=1, pin_memory=True
    )

    oe_loader = torch.utils.data.DataLoader(
        dataset=cifar100.oe_set,
        batch_size=config.batch_size // 2,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=True,
    )

    return train_loader, oe_loader, val_loader
