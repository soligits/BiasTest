from common.eval import *
import wandb
import numpy as np

model.eval()


def get_label_str(integers):
    sorted_integers = sorted(integers)
    joined_string = "-".join(map(str, sorted_integers))
    return joined_string


def log_on_wandb(results):
    try:
        wandb.log(results)
    except:
        print("Failed to Log Results on WANDB!")


if P.mode == "test_acc":
    from evals import test_classifier

    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, logger=None)

elif P.mode == "test_marginalized_acc":
    from evals import test_classifier

    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, marginal=True, logger=None)

elif P.mode in ["ood", "ood_pre"]:
    if P.mode == "ood":
        from evals import eval_ood_detection
    else:
        from evals.ood_pre import eval_ood_detection

    with torch.no_grad():
        auroc_dict = eval_ood_detection(
            P,
            model,
            test_loader,
            ood_test_loader,
            P.ood_score,
            train_loader=train_loader,
            simclr_aug=simclr_aug,
        )

    final_auroc = 0

    if P.one_class_idx is not None:
        mean_dict = dict()
        for ood_score in P.ood_score:
            mean = 0
            for ood in auroc_dict.keys():
                mean += auroc_dict[ood][ood_score]
            mean_dict[ood_score] = mean / len(auroc_dict.keys())
            final_auroc = mean_dict[ood_score]
        auroc_dict["one_class_mean"] = mean_dict

    try:
        wandb.login(key=P.wandb_api)
        wandb.init(
            # set the wandb project where this run will be logged
            project="BiasTest",
            config={
                "architecture": f"{P.model}",
                "label_count": np.unique(P.one_class_idx).shape[0],
                "dataset": P.dataset,
                "labels": get_label_str(P.one_class_idx),
            },
            tags=[
                P.dataset,
                "CSI",
            ],
        )
    except:
        print("Failed to Login to WANDB!")

    log_on_wandb({"auc": final_auroc})

    bests = []
    for ood in auroc_dict.keys():
        message = ""
        best_auroc = 0
        for ood_score, auroc in auroc_dict[ood].items():
            message += "[%s %s %.4f] " % (ood, ood_score, auroc)
            if auroc > best_auroc:
                best_auroc = auroc
        message += "[%s %s %.4f] " % (ood, "best", best_auroc)
        if P.print_score:
            print(message)
        bests.append(best_auroc)

    bests = map("{:.4f}".format, bests)
    print("\t".join(bests))

else:
    raise NotImplementedError()
