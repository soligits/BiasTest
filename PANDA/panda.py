import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
from losses import CompactnessLoss, EWCLoss
import utils
from copy import deepcopy
from tqdm import tqdm
import wandb


def log_on_wandb(results):
    try:
        wandb.log(results)
    except:
        print("Failed to Log Results on WANDB!")


def train_model(model, train_loader, test_loader, device, args, ewc_loss):
    model.eval()
    auc, feature_space = get_score(model, device, train_loader, test_loader)
    print("Epoch: {}, AUROC is: {}".format(0, auc))
    log_on_wandb({"auc": auc})

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, weight_decay=0.00005, momentum=0.9
    )
    center = torch.FloatTensor(feature_space).mean(dim=0)
    criterion = CompactnessLoss(center.to(device))
    for epoch in range(args.epochs):
        running_loss = run_epoch(
            model, train_loader, optimizer, criterion, device, args.ewc, ewc_loss
        )
        print("Epoch: {}, Loss: {}".format(epoch + 1, running_loss))
        auc, feature_space = get_score(model, device, train_loader, test_loader)
        print("Epoch: {}, AUROC is: {}".format(epoch + 1, auc))
        log_on_wandb({"train_loss": running_loss, "auc": auc})


def run_epoch(model, train_loader, optimizer, criterion, device, ewc, ewc_loss):
    running_loss = 0.0
    for i, (imgs, _) in enumerate(train_loader):
        images = imgs.to(device)

        optimizer.zero_grad()

        _, features = model(images)

        loss = criterion(features)

        if ewc:
            loss += ewc_loss(model)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / (i + 1)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for imgs, _ in tqdm(train_loader, desc="Train set feature extracting"):
            imgs = imgs.to(device)
            _, features = model(imgs)
            train_feature_space.append(features)
        train_feature_space = (
            torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
        )
    test_feature_space = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_loader, desc="Test set feature extracting"):
            imgs = imgs.to(device)
            _, features = model(imgs)
            test_feature_space.append(features)
        test_feature_space = (
            torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        )
        test_labels = test_loader.dataset.targets

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space


def get_label_str(integers):
    sorted_integers = sorted(integers)
    joined_string = "-".join(map(str, sorted_integers))
    return joined_string


def main(args):
    print(
        "Dataset: {}, Normal Label: {}, LR: {}".format(
            args.dataset, args.label, args.lr
        )
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = utils.get_resnet_model(resnet_type=args.resnet_type)
    model = model.to(device)

    ewc_loss = None

    # Freezing Pre-trained model for EWC
    if args.ewc:
        frozen_model = deepcopy(model).to(device)
        frozen_model.eval()
        utils.freeze_model(frozen_model)
        fisher = torch.load(args.diag_path)
        ewc_loss = EWCLoss(frozen_model, fisher)

    utils.freeze_parameters(model)
    train_loader, test_loader = utils.get_loaders(
        dataset=args.dataset,
        label_classes=args.label,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path,
    )
    train_model(model, train_loader, test_loader, device, args, ewc_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--dataset_path", default="~/", type=str)
    parser.add_argument(
        "--diag_path", default="./data/fisher_diagonal.pth", help="fim diagonal path"
    )
    parser.add_argument("--ewc", action="store_true", help="Train with EWC")
    parser.add_argument(
        "--epochs", default=15, type=int, metavar="epochs", help="number of epochs"
    )
    parser.add_argument("--label", type=int, help="The normal class", nargs="+")
    parser.add_argument(
        "--lr", type=float, default=1e-2, help="The initial learning rate."
    )
    parser.add_argument(
        "--resnet_type", default=152, type=int, help="which resnet to use"
    )
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--wandb_api", type=str)

    args = parser.parse_args()

    try:
        wandb.login(key=args.wandb_api)
        wandb.init(
            # set the wandb project where this run will be logged
            project="BiasTest",
            config={
                "learning_rate": args.lr,
                "architecture": f"ResNet{args.resnet_type}",
                "label_count": np.unique(args.label).shape[0],
                "dataset": args.dataset,
                "labels": get_label_str(args.label),
                "epochs": args.epochs,
            },
            tags=[
                f"ResNet{args.resnet_type}",
                args.dataset,
                "PANDA",
            ],
        )
    except:
        print("Failed to Login to WANDB!")

    main(args)
