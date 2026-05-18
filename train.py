import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold

from models.classifier import MelanomaClassifier
from data.dataset import ISICDataset
from data.augmentations import get_train_transforms, get_val_transforms
from utils.metrics import compute_metrics


def get_sampler(labels):
    # Oversample minority class (melanoma) to counter 98:2 imbalance
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, len(weights))


def train_fold(fold, train_df, val_df, image_dir, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ISICDataset(train_df, image_dir, transform=get_train_transforms())
    val_ds   = ISICDataset(val_df,   image_dir, transform=get_val_transforms())

    sampler    = get_sampler(train_df["target"].values)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=config["batch_size"] * 2,
                              shuffle=False, num_workers=4)

    model     = MelanomaClassifier(config["model_name"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                probs = torch.sigmoid(model(images.to(device))).cpu().squeeze()
                all_probs.extend(probs.numpy())
                all_labels.extend(labels.numpy())

        metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
        print(f"Fold {fold} | Epoch {epoch+1}/{config['epochs']} | "
              f"loss: {train_loss/len(train_loader):.4f} | "
              f"AUC: {metrics['auc']:.4f} | "
              f"sens: {metrics['sensitivity']:.4f}")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            torch.save(model.state_dict(), f"models/fold{fold}_best.pth")

        scheduler.step()

    print(f"Fold {fold} best AUC: {best_auc:.4f}")
    return best_auc


def main(args):
    df = pd.read_csv(args.csv)
    config = {
        "model_name": args.model,
        "epochs":     args.epochs,
        "batch_size": args.batch_size,
        "lr":         args.lr,
        "n_folds":    args.folds,
    }

    skf   = StratifiedKFold(n_splits=config["n_folds"], shuffle=True, random_state=42)
    aucs  = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["target"])):
        print(f"\n{'='*50}\nFold {fold+1}/{config['n_folds']}\n{'='*50}")
        auc = train_fold(fold+1, df.iloc[train_idx], df.iloc[val_idx], args.image_dir, config)
        aucs.append(auc)

    print(f"\nCV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    with open("cv_results.json", "w") as f:
        json.dump({"fold_aucs": aucs, "mean": np.mean(aucs), "std": np.std(aucs)}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="data/train.csv")
    parser.add_argument("--image-dir",  default="data/images/train")
    parser.add_argument("--model",      default="efficientnet_b3")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--folds",      type=int,   default=5)
    main(parser.parse_args())
