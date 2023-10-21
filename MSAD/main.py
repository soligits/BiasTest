from cProfile import label
import torch
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import argparse
import utils
from tqdm import tqdm
import torch.nn.functional as F
import gc
import logging
import numpy as np
import os
import wandb
import csv

global Logger
Logger = None


def csv_log(filename, epoch, loss, auc):
    # Opening the file in append mode
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)

        # Write header only if the file is new (i.e., empty)
        if file.tell() == 0:
            writer.writerow(["Epoch", "Loss", "AUROC"])

        writer.writerow([epoch, loss, auc])


def log(msg):
    global Logger
    Logger.write(f"{msg}\n")
    print(msg)


def contrastive_loss(out_1, out_2):
    out_1 = F.normalize(out_1, dim=-1)
    out_2 = F.normalize(out_2, dim=-1)
    bs = out_1.size(0)
    temp = 0.25
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temp)
    mask = (
        torch.ones_like(sim_matrix) - torch.eye(2 * bs, device=sim_matrix.device)
    ).bool()
    # [2B, 2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * bs, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temp)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss


def log_on_wandb(results):
    try:
        wandb.log(results)
    except:
        print("Failed to Log Results on WANDB!")


def train_model(model, train_loader, test_loader, train_loader_1, device, args):
    model.eval()

    # Ensure the directory exists
    os.makedirs("./results-csv/", exist_ok=True)

    # Generate initial filename
    file_name = f"./results-csv/MSAD-{args.dataset}-{get_label_str(args.label)}-epochs{args.epochs}-ResNet{args.backbone}.csv"
    base_name = file_name.split(".csv")[0]

    counter = 1
    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}.csv"
        counter += 1

    auc, feature_space = get_score(model, device, train_loader, test_loader)

    log_on_wandb({"auc": auc})
    log("Epoch: {}, AUROC is: {}".format(0, auc))
    csv_log(file_name, 0, "-", auc)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.00005)
    center = torch.FloatTensor(feature_space).mean(dim=0)

    if args.angular:
        center = F.normalize(center, dim=-1)

    center = center.to(device)

    for epoch in range(args.epochs):
        model.train()
        running_loss = run_epoch(
            model, train_loader_1, optimizer, center, device, args.angular
        )

        log("Epoch: {}, Loss: {}".format(epoch + 1, running_loss))

        model.eval()
        auc, _ = get_score(model, device, train_loader, test_loader)

        log("Epoch: {}, AUROC is: {}".format(epoch + 1, auc))

        csv_log(file_name, epoch + 1, running_loss, auc)
        log_on_wandb({"train_loss": running_loss, "auc": auc})
    try:
        wandb.finish()
    except:
        print("Failed to finish WANBD!")


def run_epoch(model, train_loader, optimizer, center, device, is_angular):
    total_loss, total_num = 0.0, 0
    for (img1, img2), _ in tqdm(train_loader, desc="Train..."):
        img1, img2 = img1.to(device), img2.to(device)

        optimizer.zero_grad()

        out_1 = model(img1)
        out_2 = model(img2)
        out_1 = out_1 - center
        out_2 = out_2 - center

        loss = contrastive_loss(out_1, out_2)

        if is_angular:
            loss += (out_1**2).sum(dim=1).mean() + (out_2**2).sum(dim=1).mean()

        loss.backward()

        optimizer.step()

        total_num += img1.size(0)
        total_loss += loss.item() * img1.size(0)

    return total_loss / (total_num)


def get_score(model, device, train_loader, test_loader):
    train_feature_space = []
    with torch.no_grad():
        for imgs, _ in tqdm(train_loader, desc="Train set feature extracting"):
            imgs = imgs.to(device)
            features = model(imgs)
            train_feature_space.append(features.detach().cpu())
        train_feature_space = (
            torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
        )
    test_feature_space = []
    test_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Test set feature extracting"):
            imgs = imgs.to(device)
            features = model(imgs)
            test_feature_space.append(features.detach().cpu())
            test_labels.append(labels)
        test_feature_space = (
            torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
        )
        test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

    distances = utils.knn_score(train_feature_space, test_feature_space)

    auc = roc_auc_score(test_labels, distances)

    return auc, train_feature_space


def get_label_str(integers):
    sorted_integers = sorted(integers)
    joined_string = "-".join(map(str, sorted_integers))
    return joined_string


def main(args):
    log(
        "Dataset: {}, Normal Label: {}, LR: {}".format(
            args.dataset, args.label, args.lr
        )
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(device)
    model = utils.Model(str(args.backbone))
    model = model.to(device)

    train_loader, test_loader, train_loader_1 = utils.get_loaders(
        dataset=args.dataset,
        label_classes=args.label,
        batch_size=args.batch_size,
        backbone=args.backbone,
        dataset_path=args.dataset_path,
    )

    train_model(model, train_loader, test_loader, train_loader_1, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--dataset_path", default="~/", type=str)

    parser.add_argument(
        "--epochs", default=20, type=int, metavar="epochs", help="number of epochs"
    )
    parser.add_argument("--label", type=int, help="The normal class", nargs="+")
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="The initial learning rate."
    )

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument(
        "--backbone",
        choices=["18", "50", "152", "resnet18_linf_eps8.0", "vit"],
        default="18",
        type=str,
        help="ResNet Backbone",
    )

    parser.add_argument(
        "--angular", action="store_true", help="Train with angular center loss"
    )

    parser.add_argument("--wandb_api", type=str)

    args = parser.parse_args()
    print(args)

    os.makedirs(f"./Results/", exist_ok=True)

    # Set the file name
    file_name = f"MSAD-{args.dataset}-{get_label_str(args.label)}-epochs{args.epochs}-ResNet{args.backbone}.txt"
    file_path = f"./Results/{file_name}"

    # Check if the file already exists
    if os.path.exists(file_path):
        # If it does, find a new file name by appending a number to the end
        i = 1
        while os.path.exists(f"./Results/{file_name[:-4]}_{i}.txt"):
            i += 1
        file_name = f"{file_name[:-4]}_{i}.txt"

    # Open the file for appending
    Logger = open(f"./Results/{file_name}", "a", encoding="utf-8")

    try:
        wandb.login(key=args.wandb_api)
        wandb.init(
            # set the wandb project where this run will be logged
            project="BiasTest",
            config={
                "learning_rate": args.lr,
                "architecture": f"ResNet{args.backbone}",
                "label_count": np.unique(args.label).shape[0],
                "dataset": args.dataset,
                "labels": get_label_str(args.label),
                "epochs": args.epochs,
            },
            tags=[
                f"ResNet{args.backbone}",
                args.dataset,
                "MSAD" if args.epochs > 0 else "DN2",
            ],
        )
    except:
        log("Failed to Login to WANDB!")

    main(args)
