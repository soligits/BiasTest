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
import torch.nn as nn
import pandas as pd

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


def log_on_wandb(results):
    try:
        wandb.log(results)
    except:
        print("Failed to Log Results on WANDB!")


def append_auc_to_csv(auc_dict, csv_path):
    # Convert AUCs to DataFrame
    df_auc = pd.DataFrame([auc_dict])
    # Check if the CSV file exists
    if os.path.exists(csv_path):
        df_auc.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_auc.to_csv(csv_path, mode='w', header=True, index=False)

def calculate_average_auc(train_loader, test_loader, model, device, csv_path):
    aucs = []
    auc_dict = {}
    # Check if test_loader is a list of datasets
    if isinstance(test_loader, list):
        for test_set in test_loader:
            auc = get_score(model, device, test_set)
            dataset_name = str(test_set.dataset)
            aucs.append(auc)
            auc_dict[dataset_name] = auc
            print("AUC for {}: {}".format(dataset_name, auc))
    else:  # If it's a single dataset
        auc = get_score(model, device, test_loader)
        aucs.append(auc)
        auc_dict['single_testset'] = auc
        print("AUC for the single test set: {}".format(auc))

    # Append AUCs to CSV
    append_auc_to_csv(auc_dict, csv_path)
    return np.mean(aucs)

def train_model(model, train_loader, test_loader, device, args):
    model.eval()

    # Ensure the directory exists
    os.makedirs("./results-csv/", exist_ok=True)

    # Generate initial filename
    file_name = f"./results-csv/RotNet-{args.dataset}-{get_label_str(args.label)}-epochs{args.epochs}-ResNet{args.backbone}.csv"
    base_name = file_name.split(".csv")[0]

    counter = 1
    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}.csv"
        counter += 1

    model.eval()
    auc = calculate_average_auc(train_loader, test_loader, model, device, f'{args.dataset}_{get_label_str(args.label)}_auc.csv')
    print("Epoch: 0, AUROC is: {}".format(auc))

    log_on_wandb({"auc": auc})
    
    csv_log(file_name, 0, "-", auc)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running_loss = run_epoch(model, train_loader, optimizer, criterion, device)

        log("Epoch: {}, Loss: {}".format(epoch + 1, running_loss))

    auc = calculate_average_auc(train_loader, test_loader, model, device, f'{args.dataset}_{get_label_str(args.label)}_auc.csv')
    print("Epoch: {}, AUROC is: {}".format(epoch + 1, auc))
    
    log_on_wandb({"auc": auc})
    
    try:
        wandb.finish()
    except:
        print("Failed to finish WANBD!")
    
def run_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_num = 0
    for data in tqdm(train_loader, desc="Train..."):
        inputs_original, labels = data[0].to(device), data[1].to(device)

        # Use with torch.no_grad() for operations that don't need gradient tracking
        with torch.no_grad():
            # Make 4 variations of input with different degrees of rotation
            inputs_90 = torch.rot90(inputs_original, 1, [2, 3])
            inputs_180 = torch.rot90(inputs_original, 2, [2, 3])
            inputs_270 = torch.rot90(inputs_original, 3, [2, 3])
            inputs = torch.cat((inputs_original, inputs_90, inputs_180, inputs_270), 0)

        labels = torch.cat(
            (torch.zeros(len(inputs_original), device=device),
             torch.ones(len(inputs_original), device=device),
             torch.full((len(inputs_original),), 2, device=device),
             torch.full((len(inputs_original),), 3, device=device)),
            0
        ).long()

        # Zero the gradients before the forward pass
        optimizer.zero_grad()
        
        # Forward pass
        rot_pred = model(inputs)
        loss = criterion(rot_pred, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Use item() to get a Python float and release the graph
        running_loss += loss.item() * inputs.size(0)
        total_num += inputs.size(0)

        # Detach the variables to break from the current graph and save memory
        inputs = inputs.detach()
        labels = labels.detach()

    # Calculate the average loss
    average_loss = running_loss / total_num

    # Clearing up memory at the end of the epoch
    torch.cuda.empty_cache()

    return average_loss


def get_score(model, device, test_loader):
    model.eval()
    anomaly_scores = []
    test_labels = []
    with torch.no_grad():  # Disable gradient tracking
        for i, (inputs_original, labels) in tqdm(enumerate(test_loader), desc="Testing"):
            inputs_original, labels = inputs_original.to(device), labels.to(device)

            # Make 4 variations of input with different degrees of rotation
            inputs_90 = torch.rot90(inputs_original, 1, [2, 3])
            inputs_180 = torch.rot90(inputs_original, 2, [2, 3])
            inputs_270 = torch.rot90(inputs_original, 3, [2, 3])
            inputs = torch.cat((inputs_original, inputs_90, inputs_180, inputs_270), 0)

            # Forward pass
            rot_pred = model(inputs)

            c = inputs_original.shape[0]
            rot_loss = torch.zeros(c, device=device)
            for j in range(4):  # Calculate the loss for each rotation
                rot_loss += F.log_softmax(rot_pred[j * c : (j + 1) * c], dim=1)[:, j]

            # Append scores and labels for AUROC calculation
            anomaly_scores.append(rot_loss.cpu())  # Move to CPU before concatenating
            test_labels.append(labels.cpu())

    # Concatenate all the scores and labels
    anomaly_scores = torch.cat(anomaly_scores).numpy()
    test_labels = torch.cat(test_labels).numpy()

    # Calculate AUROC
    auroc = roc_auc_score(test_labels, anomaly_scores)
    
    # Clearing up memory after testing
    torch.cuda.empty_cache()
    model.train()

    return auroc

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

    train_loader, test_loader = utils.get_loaders(
        dataset=args.dataset,
        label_classes=args.label,
        batch_size=args.batch_size,
        dataset_path=args.dataset_path,
    )

    train_model(model, train_loader, test_loader, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--dataset_path", default="~/", type=str)

    parser.add_argument(
        "--epochs", default=5, type=int, metavar="epochs", help="number of epochs"
    )
    parser.add_argument("--label", type=int, help="The normal class", nargs="+")
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="The initial learning rate."
    )

    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument(
        "--backbone",
        choices=[
            "18",
            "50",
            "152",
        ],
        default="18",
        type=str,
        help="ResNet Backbone",
    )

    parser.add_argument("--wandb_api", type=str)

    args = parser.parse_args()
    print(args)

    os.makedirs(f"./Results/", exist_ok=True)

    # Set the file name
    file_name = f"RotNet-{args.dataset}-{get_label_str(args.label)}-epochs{args.epochs}-ResNet{args.backbone}.txt"
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
            project="RotNet-Corrupt",
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
                "RotNet",
            ],
        )
    except:
        log("Failed to Login to WANDB!")

    main(args)
