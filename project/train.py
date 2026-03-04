import argparse
import copy
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from calibration import apply_temperature, expected_calibration_error, fit_temperature, plot_reliability_diagram
from ensemble import collect_ensemble_probabilities, disagreement_from_ensemble_probs
from models import MCDropoutResNet18, load_checkpoint, save_checkpoint
from trust_metrics import classify_trust_levels, compute_ensemble_disagreement, compute_trust_score
from utils import (
    binary_classification_metrics,
    get_dataloaders,
    get_device,
    optimize_binary_threshold,
    plot_distribution,
    plot_roc,
    save_json,
    set_seed,
)


# ------------------------------
# Training and Evaluation
# ------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    all_logits = []
    all_probs = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)[:, 1]

        running_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach().cpu())
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_logits = torch.cat(all_logits, dim=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = binary_classification_metrics(all_labels, all_probs)
    metrics["loss"] = running_loss / len(loader.dataset)

    return metrics, all_logits, all_probs, all_labels


@torch.no_grad()
def mc_dropout_predict(model, loader, device, passes=30):
    """Run MC Dropout across dataloader and return mean probabilities + uncertainty."""
    all_mc_probs = []
    all_labels = []

    model.eval()

    for images, labels in tqdm(loader, desc="MC Dropout", leave=False):
        images = images.to(device)
        batch_probs = []

        for _ in range(passes):
            # Enable dropout at inference by setting dropout modules to train
            for module in model.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    module.train()

            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1]
            batch_probs.append(probs.detach().cpu().numpy())

        batch_probs = np.stack(batch_probs, axis=0)  # (passes, batch)
        all_mc_probs.append(batch_probs)
        all_labels.append(labels.numpy())

    all_mc_probs = np.concatenate(all_mc_probs, axis=1)  # (passes, N)
    all_labels = np.concatenate(all_labels, axis=0)

    mean_probs = all_mc_probs.mean(axis=0)
    uncertainty = all_mc_probs.var(axis=0)
    return mean_probs, uncertainty, all_mc_probs, all_labels


# ------------------------------
# Base Model Training
# ------------------------------
def train_base_model(args, train_loader, val_loader, class_names, device, seed=42):
    set_seed(seed)

    model = MCDropoutResNet18(
        num_classes=len(class_names),
        dropout_p=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"TrainLoss={train_loss:.4f} "
            f"ValLoss={val_metrics['loss']:.4f} "
            f"ValAcc={val_metrics['accuracy']:.4f} "
            f"ValF1={val_metrics['f1']:.4f} "
            f"ValAUC={val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    os.makedirs(args.output_dir, exist_ok=True)
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    save_checkpoint(
        best_model_path,
        model,
        extra={
            "class_names": class_names,
            "dropout": args.dropout,
            "best_val_auc": float(best_auc),
            "seed": seed,
        },
    )

    print(f"Saved best model to: {best_model_path}")
    return model, best_model_path


# ------------------------------
# Ensemble Training
# ------------------------------
def train_ensemble(args, train_loader, val_loader, class_names, device):
    seeds = [42, 123, 999]
    ensemble_paths = []
    ensemble_tmp_root = os.path.join(args.output_dir, "ensemble_tmp")
    os.makedirs(ensemble_tmp_root, exist_ok=True)

    for idx, seed in enumerate(seeds, start=1):
        print(f"\nTraining ensemble model {idx}/{len(seeds)} with seed={seed}")
        local_args = copy.deepcopy(args)
        local_args.epochs = args.ensemble_epochs
        local_args.output_dir = os.path.join(ensemble_tmp_root, f"seed_{seed}")
        model, model_path = train_base_model(local_args, train_loader, val_loader, class_names, device, seed=seed)

        # Rename to keep all 3 checkpoints
        target_path = os.path.join(args.output_dir, f"ensemble_model_seed_{seed}.pt")
        shutil.copy2(model_path, target_path)
        ensemble_paths.append(target_path)

    return ensemble_paths


# ------------------------------
# Main Research Pipeline
# ------------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    set_seed(args.seed)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )
    print(f"Class names: {class_names}")

    # 1) Base training
    model, best_model_path = train_base_model(args, train_loader, val_loader, class_names, device, seed=args.seed)

    criterion = nn.CrossEntropyLoss()
    test_metrics, test_logits, test_probs, test_labels = evaluate(model, test_loader, criterion, device)

    # 2) MC Dropout uncertainty (30 passes)
    mc_mean_probs, mc_uncertainty, mc_all_probs, _ = mc_dropout_predict(model, test_loader, device, passes=30)

    # 3) Calibration analysis (ECE + temperature scaling)
    _, val_logits, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
    ece_before = expected_calibration_error(val_probs, val_labels, n_bins=args.ece_bins)

    temperature = fit_temperature(val_logits, torch.tensor(val_labels, dtype=torch.long), device=device)

    # Apply temperature scaling to test logits
    scaled_test_logits = apply_temperature(test_logits, temperature)
    scaled_test_probs = F.softmax(scaled_test_logits, dim=1)[:, 1].detach().cpu().numpy()

    scaled_val_logits = apply_temperature(val_logits, temperature)
    scaled_val_probs = F.softmax(scaled_val_logits, dim=1)[:, 1].detach().cpu().numpy()
    best_threshold, best_val_f1 = optimize_binary_threshold(val_labels, scaled_val_probs)

    ece_after = expected_calibration_error(scaled_test_probs, test_labels, n_bins=args.ece_bins)

    plot_reliability_diagram(
        test_probs,
        test_labels,
        n_bins=args.ece_bins,
        title="Reliability Diagram (Before Temperature Scaling)",
        save_path=os.path.join(figures_dir, "reliability_before.png"),
    )
    plot_reliability_diagram(
        scaled_test_probs,
        test_labels,
        n_bins=args.ece_bins,
        title="Reliability Diagram (After Temperature Scaling)",
        save_path=os.path.join(figures_dir, "reliability_after.png"),
    )

    # 4) Ensemble disagreement (3 models, different seeds)
    ensemble_paths = train_ensemble(args, train_loader, val_loader, class_names, device)
    ensemble_models = []
    for p in ensemble_paths:
        m = MCDropoutResNet18(
            num_classes=len(class_names),
            dropout_p=args.dropout,
            pretrained=False,
        ).to(device)
        load_checkpoint(p, m, map_location=device)
        m.eval()
        ensemble_models.append(m)

    ensemble_probs, _ = collect_ensemble_probabilities(ensemble_models, test_loader, device)
    disagreement = disagreement_from_ensemble_probs(ensemble_probs)

    # Keep a named alias consistent with requirement text
    _ = compute_ensemble_disagreement(ensemble_probs)

    # 5-6) Clinical risk weighting + final trust score
    confidence = scaled_test_probs
    pred_labels = (scaled_test_probs >= best_threshold).astype(int)

    weights = {
        "uncertainty": args.w1,
        "ece": args.w2,
        "disagreement": args.w3,
        "confidence": args.w4,
    }

    trust_scores = compute_trust_score(
        uncertainty=mc_uncertainty,
        ece=ece_after,
        disagreement=disagreement,
        confidence=confidence,
        pred_labels=pred_labels,
        class_names=class_names,
        weights=weights,
        risk_weights={"normal": 1.0, "pneumonia": 1.5},
    )

    trust_levels = classify_trust_levels(
        trust_scores,
        pred_labels,
        class_names,
        reliable_threshold=args.reliable_threshold,
        review_threshold=args.review_threshold,
        high_risk_margin=args.high_risk_margin,
    )

    # 7) Visualization outputs
    plot_distribution(
        mc_uncertainty,
        title="MC Dropout Uncertainty Distribution",
        xlabel="Uncertainty (Variance)",
        save_path=os.path.join(figures_dir, "uncertainty_distribution.png"),
    )
    plot_distribution(
        trust_scores,
        title="Trust Score Distribution",
        xlabel="Trust Score",
        save_path=os.path.join(figures_dir, "trust_distribution.png"),
    )
    plot_roc(test_labels, scaled_test_probs, save_path=os.path.join(figures_dir, "roc_curve.png"))

    # Save per-sample trust analysis table
    report_df = pd.DataFrame(
        {
            "label": test_labels,
            "probability": scaled_test_probs,
            "confidence": confidence,
            "uncertainty": mc_uncertainty,
            "disagreement": disagreement,
            "trust_score": trust_scores,
            "trust_level": trust_levels,
        }
    )
    report_csv = os.path.join(args.output_dir, "trust_analysis.csv")
    report_df.to_csv(report_csv, index=False)

    # 10) Final research output
    scaled_metrics = binary_classification_metrics(test_labels, scaled_test_probs, threshold=best_threshold)
    final_metrics = {
        "accuracy": scaled_metrics["accuracy"],
        "auc": scaled_metrics["auc"],
        "f1": scaled_metrics["f1"],
        "ece_before": ece_before,
        "ece_after": ece_after,
        "brier": scaled_metrics["brier"],
        "avg_trust_score": float(np.mean(trust_scores)),
        "temperature": float(temperature),
        "decision_threshold": float(best_threshold),
        "best_val_f1": float(best_val_f1),
        "class_names": class_names,
        "best_model_path": best_model_path,
        "ensemble_model_paths": ensemble_paths,
    }

    summary_path = os.path.join(args.output_dir, "research_summary.json")
    save_json(final_metrics, summary_path)

    print("\n========== RESEARCH OUTPUT ==========")
    print(f"Final accuracy      : {final_metrics['accuracy']:.4f}")
    print(f"AUC                 : {final_metrics['auc']:.4f}")
    print(f"F1                  : {final_metrics['f1']:.4f}")
    print(f"ECE (before)        : {final_metrics['ece_before']:.4f}")
    print(f"ECE (after)         : {final_metrics['ece_after']:.4f}")
    print(f"Brier score         : {final_metrics['brier']:.4f}")
    print(f"Average trust score : {final_metrics['avg_trust_score']:.4f}")
    print(f"Decision threshold  : {final_metrics['decision_threshold']:.2f}")
    print(f"Saved summary to    : {summary_path}")
    print(f"Saved report to     : {report_csv}")
    print(f"Saved figures in    : {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reliability and Trust Metrics for Chest X-ray Classification")

    # Dataset and runtime
    parser.add_argument("--data_root", type=str, default="/content/chest_xray", help="Path containing train/val/test or train/test folders")
    parser.add_argument("--output_dir", type=str, default="./artifacts")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.15)

    # Training
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--ensemble_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretrained weights")

    # Calibration and trust
    parser.add_argument("--ece_bins", type=int, default=15)
    parser.add_argument("--w1", type=float, default=0.30, help="Weight for uncertainty")
    parser.add_argument("--w2", type=float, default=0.20, help="Weight for ECE")
    parser.add_argument("--w3", type=float, default=0.20, help="Weight for disagreement")
    parser.add_argument("--w4", type=float, default=0.30, help="Weight for confidence")
    parser.add_argument("--reliable_threshold", type=float, default=0.72)
    parser.add_argument("--review_threshold", type=float, default=0.50)
    parser.add_argument("--high_risk_margin", type=float, default=0.08)

    args = parser.parse_args()
    main(args)
