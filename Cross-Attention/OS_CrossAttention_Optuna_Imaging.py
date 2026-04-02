#!/usr/bin/env python
"""
Optuna-based Hyperparameter Optimization for Imaging-only Osteosarcoma model.
Adapted to use Optuna and main_tuning framework structure with OS dataset.

Key changes:
1. Replaced SMAC3 with Optuna for hyperparameter search
2. Uses predefined train/test splits (similar to main_tuning.py)
3. Implements Optuna suggest functions for hyperparameters
4. Uses OS dataset with PyTorch Lightning training
5. Includes 5-fold inner cross-validation per outer fold
"""

import sys
import os
import json
import copy
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict, Counter
from pathlib import Path

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.metrics import roc_auc_score
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedKFold

from fuse.dl.models.model_multihead import ModelMultiHead
from fuse.dl.losses.loss_default import LossDefault
from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAUCROC, MetricConfusion, MetricBSS, MetricAccuracy
)
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds

import torch.optim as optim
import pytorch_lightning as pl
from fuse.dl.lightning.pl_module import LightningModuleDefault
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt

from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault

from OS import OSDataset
from HeadMLPClassifier import HeadMLPClassifier
from x_transformers import Encoder, TransformerWrapper

# =============================
# Global constants
# =============================

RUNS_DIR_NAME = "runs_optuna_OS"

OPTUNA_STUDY_NAME = "os_imaging_optuna"

KEY_PROB = "model.prob.NAC_Classification"
KEY_LOGITS = "model.logits.NAC_Classification"
KEY_TARGET = "data.input.clinical.raw.Huvosnew"
KEY_TARGET_F = "data.input.clinical.raw.Huvosnew.f"
KEY_SAMPLE_ID = "data.sample_id"

REPORT_KEYS = [
    "patch_size", "emb_dim",
    "lr", "wd",
    "depth_a", "heads_a",
    "depth_b", "heads_b",
    "depth_cross_attn", "heads_cross",
    "mlp_layers",
    "mask_pad_thresh", "clinical_aug", "imaging_aug_deg",
    "batch_size", "num_workers",
]

# =============================
# Utils
# =============================

def seed_everything(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ceil_div(a, b):
    return (a + b - 1) // b


def compute_max_num_tokens(largest_tumor: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> int:
    zc = ceil_div(largest_tumor[0], patch_size[0]) 
    yc = ceil_div(largest_tumor[1], patch_size[1])
    xc = ceil_div(largest_tumor[2], patch_size[2])
    return int(zc * yc * xc)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def config_slug(cfg: Dict[str, Any], max_len: int = 120) -> str:
    import hashlib, re
    ABBR = {
        "patch_size": "ps", "emb_dim": "ed", "lr": "lr", "wd": "wd",
        "depth_a": "da", "heads_a": "ha", "depth_b": "db", "heads_b": "hb",
        "depth_cross_attn": "dca", "heads_cross": "hc",
        "mlp_layers": "mlp", "mask_pad_thresh": "mpt",
        "clinical_aug": "ca", "imaging_aug_deg": "ia",
    }
    ORDER = [
        "patch_size","emb_dim","lr","wd",
        "depth_a","heads_a","depth_b","heads_b","depth_cross_attn","heads_cross",
        "mlp_layers","mask_pad_thresh","clinical_aug","imaging_aug_deg"
    ]
    def fmt_lr(x: float) -> str:
        if float(x) == 0.0:
            return "0"
        s = np.format_float_scientific(float(x), precision=0, unique=True, exp_digits=1)
        return s.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    def fmt_ps(ps) -> str:
        return "x".join(str(int(v)) for v in ps)
    def fmt_mlp(v: str) -> str:
        return "S" if str(v).lower().startswith("s") else "D"
    def fmt_pct10(v: float) -> str:
        return f"{int(round(float(v)*10)):02d}"
    def clean(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9\-x]", "", s)

    parts = []
    for k in ORDER:
        if k not in cfg:
            continue
        ab = ABBR[k]
        v = cfg[k]
        if k == "patch_size":
            val = fmt_ps(v)
        elif k in ("emb_dim","depth_a","heads_a","depth_b","heads_b","depth_cross_attn","heads_cross","imaging_aug_deg"):
            val = str(int(v))
        elif k in ("lr","wd"):
            val = fmt_lr(v)
        elif k == "mlp_layers":
            val = fmt_mlp(v)
        elif k in ("mask_pad_thresh","clinical_aug"):
            val = fmt_pct10(v)
        else:
            val = str(v)
        parts.append(f"{ab}{clean(val)}")
    slug = "-".join(parts)
    if len(slug) > max_len:
        cfg_subset = {k: cfg.get(k) for k in ORDER if k in cfg}
        h = hashlib.sha1(json.dumps(cfg_subset, sort_keys=True, default=str).encode()).hexdigest()[:8]
        slug = slug[: max_len - (len(h) + 2)] + "-h" + h
    return slug


# =============================
# Imaging-only backbone (same as SMAC version)
# =============================

class ImagingTransformerEncoder(nn.Module):
    """
    Imaging-only backbone using TransformerWrapper + Encoder on tumor patch tokens.

    Inputs via ModelMultiHead.backbone_args:
        xb           <- batch_dict['data.input.img.tumor3d.patches']   [B, N_p, patch_dim]
        embed_mask_b <- batch_dict['model.embed_mask_b']               [B, N_p] (bool)

    Output:
        [B, N_p, D] token embeddings
    """
    def __init__(
        self,
        emb_dim: int,
        depth_b: int,
        heads_b: int,
        patch_dim: int,
        max_seq_len_b: int,
    ):
        super().__init__()

        token_emb = nn.Linear(patch_dim, emb_dim)

        kwargs_wrapper_b = {
            'token_emb': token_emb,
            'use_abs_pos_emb': False,
            'emb_dropout': 0.1,
            'post_emb_norm': True,
            'return_only_embed': True,
        }
        kwargs_encoder_b = {
            'attn_dropout': 0.1,
            'ff_dropout': 0.1,
            'ff_glu': False,
            'use_rmsnorm': True,
            'rotary_pos_emb': True,
        }

        self.enc_b = TransformerWrapper(
            num_tokens=None,
            max_seq_len=max_seq_len_b,
            **kwargs_wrapper_b,
            attn_layers=Encoder(
                dim=emb_dim,
                depth=depth_b,
                heads=heads_b,
                **kwargs_encoder_b,
            ),
        )

    def forward(
        self,
        xb: torch.Tensor,
        embed_mask_b: torch.Tensor,
    ) -> torch.Tensor:
        enc_xb = self.enc_b(
            xb,
            return_embeddings=True,
            mask=embed_mask_b
        )
        return enc_xb


def build_model(
    cfg: Dict[str, Any],
    patch_dim: int,
    max_num_tokens_b: int,
) -> ModelMultiHead:
    """
    Imaging-only model:
    - ImagingTransformerEncoder backbone (sequence of CT patch tokens)
    - Attention pooling + MLP classifier head
    """
    if cfg.get("mlp_layers", "single") == "double":
        layers_desc = (cfg["emb_dim"], max(1, cfg["emb_dim"] // 2))
    else:
        layers_desc = (cfg["emb_dim"],)

    backbone = ImagingTransformerEncoder(
        emb_dim=cfg["emb_dim"],
        depth_b=cfg.get("depth_b", 2),
        heads_b=cfg.get("heads_b", 4),
        patch_dim=patch_dim,
        max_seq_len_b=max_num_tokens_b,
    )

    model = ModelMultiHead(
        backbone=backbone,
        key_out_features='model.backbone_features',
        backbone_args=[
            'data.input.img.tumor3d.patches',  # xb
            'model.embed_mask_b',             # mask for xb
        ],

        heads=[
            HeadMLPClassifier(
                input_key='model.backbone_features',
                prob_key=KEY_PROB,
                logits_key=KEY_LOGITS,
                in_ch=cfg["emb_dim"],
                num_classes=1,
                layers_description=layers_desc,
                dropout_rate=0.1,
                pooling="attention",
            )
        ],
    )
    return model
class SimpleLossPlotCallback(Callback):
    """Lightweight callback for real-time loss plotting.

    Captures losses directly from trainer.callback_metrics to avoid CSV flush
    timing issues (the CSVLogger may not have written the current epoch's
    training rows to disk yet when on_validation_end fires).
    """
    def __init__(self, save_every_n_epochs: int = 5):
        super().__init__()
        self.save_every_n_epochs = save_every_n_epochs
        self._train_losses: dict = {}  # epoch -> loss
        self._val_losses: dict = {}    # epoch -> loss

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        if 'train.losses.total_loss' in metrics:
            self._train_losses[epoch] = float(metrics['train.losses.total_loss'])

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        if 'validation.losses.total_loss' in metrics:
            self._val_losses[epoch] = float(metrics['validation.losses.total_loss'])

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.save_every_n_epochs == 0:
            try:
                log_dir = trainer.log_dir if hasattr(trainer, 'log_dir') else trainer.default_root_dir
                if log_dir:
                    self._plot(log_dir, current_epoch)
            except Exception:
                pass

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Always save a final plot after training so all epochs are included."""
        try:
            log_dir = trainer.log_dir if hasattr(trainer, 'log_dir') else trainer.default_root_dir
            if log_dir:
                self._plot(log_dir, trainer.current_epoch)
        except Exception:
            pass

    def _plot(self, log_dir: str, current_epoch: int) -> None:
        try:
            plt.figure(figsize=(12, 7))

            if self._train_losses:
                epochs = sorted(self._train_losses)
                losses = [self._train_losses[e] for e in epochs]
                plt.plot(epochs, losses, label='Training Loss',
                         linewidth=2.5, color='#1f77b4', marker='o', markersize=4)
                print(f"[PLOT] Training losses plotted: {len(epochs)} epochs")

            if self._val_losses:
                epochs = sorted(self._val_losses)
                losses = [self._val_losses[e] for e in epochs]
                plt.plot(epochs, losses, label='Validation Loss',
                         linewidth=2.5, color='#ff7f0e', marker='s', markersize=4)
                print(f"[PLOT] Validation losses plotted: {len(epochs)} epochs")

            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.title(f'Loss Curves (Updated at Epoch {current_epoch})', fontsize=14)
            plt.legend(fontsize=11, loc='best')
            plt.grid(True, alpha=0.3)

            plot_path = os.path.join(log_dir, "loss_curves_realtime.png")
            plt.savefig(plot_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"[PLOT] Saved plot to {plot_path}")
        except Exception as e:
            print(f"[WARN] Failed to create realtime plot: {e}")
            import traceback
            traceback.print_exc()


def make_training_elements():
    losses = {
        "OS_cls_loss": LossDefault(
            pred=KEY_LOGITS,
            target=KEY_TARGET_F,
            callable=lambda pred, target: F.binary_cross_entropy_with_logits(pred, target.unsqueeze(1)),
            weight=1.0
        )
    }
    confusion_metrics = ['sensitivity', 'specificity', 'ppv', 'f1']
    common_metrics = OrderedDict([
        ("auc", MetricAUCROC(pred=KEY_PROB, target=KEY_TARGET)),
        ("apply_thresh", MetricApplyThresholds(pred=KEY_PROB, operation_point=0.5)),
        ("confusion", MetricConfusion(
            pred="results:metrics.apply_thresh.cls_pred",
            target=KEY_TARGET,
            metrics=confusion_metrics
        )),
        ("accuracy", MetricAccuracy(
            pred="results:metrics.apply_thresh.cls_pred",
            target=KEY_TARGET
        )),
        # ("bss", MetricBSS(pred=KEY_PROB, target=KEY_TARGET)),
    ])
    train_metrics = common_metrics.copy()
    validation_metrics = copy.deepcopy(common_metrics)
    best_epoch_source = dict(monitor="validation.metrics.auc", mode="max")
    return losses, train_metrics, validation_metrics, best_epoch_source


# =============================
# Data helpers
# =============================

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def build_dataloaders_from_ids(
    data_paths: Dict[str, str],
    cfg: Dict[str, Any],
    largest_tumor: Tuple[int, int, int],
    train_ids: List[str],
    val_ids: List[str],
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train & val dataloaders from explicit sample ID lists.
    Uses the OS dataset with configurable augmentation.
    """
    patch_size = cfg["patch_size"]
    angle_range = (-cfg["imaging_aug_deg"], cfg["imaging_aug_deg"])
    mask_pad_threshold = cfg["mask_pad_thresh"]
    dropout_p = cfg["clinical_aug"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    train_dataset = OSDataset.dataset(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        train=True,
        sample_ids=train_ids,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        mask_pad_threshold=mask_pad_threshold,
        dropout_p=dropout_p,
    )

    val_dataset = OSDataset.dataset(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        train=False,
        sample_ids=val_ids,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        mask_pad_threshold=mask_pad_threshold,
        dropout_p=dropout_p,
    )

    # Create DataLoaders with safe collate function
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=CollateDefault(),
        num_workers=0,  # Start with 0 workers for stability
        pin_memory=False,  # Disable for debugging
        shuffle=True,
        drop_last=False,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=CollateDefault(),
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
    )

    return train_dataloader, val_dataloader


# =============================
# Optuna: Hyperparameter suggestion (replacing SMAC ConfigSpace)
# =============================

def suggest_hyperparameters(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suggest hyperparameters for Optuna trial, replacing the SMAC ConfigSpace approach.
    """
    cfg = copy.deepcopy(base_cfg)
    
    # Patch size - use string representation to avoid Optuna warnings
    patch_size_options = [
        "12x48x48",  # String representation
        "16x32x32",
        "16x64x64"
    ]
    patch_size_str = trial.suggest_categorical("patch_size", patch_size_options)
    
    # Convert string back to tuple
    cfg["patch_size"] = tuple(map(int, patch_size_str.split('x')))
    
    # Embedding dimension
    cfg["emb_dim"] = trial.suggest_categorical("emb_dim", [64, 128])
    
    # Learning rate and weight decay
    cfg["lr"] = trial.suggest_float("lr", 3e-5, 5e-4, log=True)
    cfg["wd"] = trial.suggest_float("wd", 1e-6, 5e-3, log=True)
    
    # Transformer encoder depths and heads
    cfg["depth_b"] = trial.suggest_categorical("depth_b", [2, 3])
    cfg["heads_b"] = trial.suggest_categorical("heads_b", [4, 8])
    
    # MLP layers
    cfg["mlp_layers"] = trial.suggest_categorical("mlp_layers", ["double"])
    
    # Augmentation and masking
    cfg["mask_pad_thresh"] = trial.suggest_categorical("mask_pad_thresh", [0.7, 0.8])
    cfg["imaging_aug_deg"] = trial.suggest_categorical("imaging_aug_deg", [0, 10, 20])
    
    # DataLoader parameters (add these if not in base_cfg)
    cfg["num_workers"] = 10  # Keep at 0 for stability
    
    return cfg


def reconstruct_cfg_from_params(params: Dict[str, Any], base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebuild a full config dict from a completed Optuna trial's params dict.
    Keys not in params fall back to base_cfg values.
    """
    cfg = copy.deepcopy(base_cfg)
    param_map = {
        "emb_dim", "lr", "wd", "depth_b", "heads_b",
        "mlp_layers", "mask_pad_thresh", "imaging_aug_deg",
    }
    for key in param_map:
        if key in params:
            cfg[key] = params[key]
    if "patch_size" in params:
        cfg["patch_size"] = tuple(map(int, params["patch_size"].split("x")))
    cfg["num_workers"] = 10
    return cfg


# =============================
# Training a single trial
# =============================

def train_one_trial(
    run_dir: str,
    cfg: Dict[str, Any],
    data_paths: Dict[str, str],
    largest_tumor: Tuple[int, int, int],
    train_ids: List[str],
    val_ids: List[str],
    seed: int,
) -> float:
    """
    Train an imaging-only model for a single config on a single CV fold and single seed.
    Returns the validation AUC (float) based on best epoch.
    """
    ensure_dir(run_dir)
    seed_everything(seed)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    patch_size = cfg["patch_size"]
    patch_dim = int(np.prod(patch_size))
    max_num_tokens_b = compute_max_num_tokens(largest_tumor, patch_size)

    train_dl, val_dl = build_dataloaders_from_ids(
        data_paths=data_paths,
        cfg=cfg,
        largest_tumor=largest_tumor,
        train_ids=train_ids,
        val_ids=val_ids,
    )

    model = build_model(cfg, patch_dim, max_num_tokens_b)
    losses, train_metrics, validation_metrics, best_epoch_source = make_training_elements()

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=(0.5 ** 4) * cfg["lr"],
    )
    optimizers_and_lr_sch = dict(
        optimizer=optimizer,
        lr_scheduler=dict(scheduler=lr_scheduler, monitor="validation.metrics.auc")
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=run_dir,
        monitor="validation.metrics.auc",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best_epoch",
        auto_insert_metric_name=False,
        verbose=True
    )

    early_stop_cb = EarlyStopping(
        monitor="validation.metrics.auc",
        patience=5, 
        mode="max",
        verbose=True
    )
    csv_logger = CSVLogger(save_dir=run_dir, name=".")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    loss_plot_cb = SimpleLossPlotCallback(save_every_n_epochs=1)

    pl_module = LightningModuleDefault(
        model_dir=run_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_sch,
    )

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=100, 
        accelerator="auto",
        devices=1,
        logger=csv_logger,
        log_every_n_steps=1,  # Disable step-level logging
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        callbacks=[early_stop_cb, checkpoint_cb, lr_monitor, loss_plot_cb],
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    trainer.fit(pl_module, train_dl, val_dl)

    # Replace the entire try-except block with:
    try:
        log_dir = csv_logger.log_dir
        src_metrics = os.path.join(log_dir, "metrics.csv")
        dst_metrics = os.path.join(run_dir, "metrics.csv")
        
        if os.path.exists(src_metrics):
            # Read and clean the CSV
            df = pd.read_csv(src_metrics)
            
            # Keep only epoch-based rows (where epoch is not NaN)
            df_clean = df.dropna(subset=['epoch']).copy()
            
            # If we have epoch data, save cleaned version
            if len(df_clean) > 0:
                # Take only the last step of each epoch to avoid duplicates
                df_clean = df_clean.groupby('epoch').last().reset_index()
                df_clean.to_csv(dst_metrics, index=False)
                print(f"[INFO] Saved cleaned metrics with {len(df_clean)} epochs to {dst_metrics}")
            else:
                # If no epoch data, just copy as-is
                shutil.copy2(src_metrics, dst_metrics)
            
            # Save a separate CSV with all steps (for debugging)
            df.to_csv(os.path.join(run_dir, "metrics_all_steps.csv"), index=False)

            # copy the loss curves plot
            src_plot = os.path.join(log_dir, "loss_curves_realtime.png")
            dst_plot = os.path.join(run_dir, "loss_curves.png")
            if os.path.exists(src_plot):
                shutil.copy2(src_plot, dst_plot)
        
        # Clean up log directory
        try:
            shutil.rmtree(log_dir)
        except Exception as e:
            print(f"[WARN] Could not remove log directory {log_dir}: {e}")
            
    except Exception as e:
        print(f"[WARN] Failed to process logs for {run_dir}: {e}")

    # Determine best validation AUC from metrics.csv
    val_auc = 0.0
    metrics_path = os.path.join(run_dir, "metrics.csv")

    if os.path.exists(metrics_path):
        try:
            df = pd.read_csv(metrics_path)
            if "validation.metrics.auc" in df.columns:
                values = df["validation.metrics.auc"].dropna()
                if len(values) > 0:
                    val_auc = float(values.max())
                    print(f"[INFO] Using best val_auc={val_auc:.4f} from metrics.csv")
                else:
                    print(f"[WARN] validation.metrics.auc column empty in {metrics_path}")
            else:
                print(f"[WARN] validation.metrics.auc not found in {metrics_path}")
        except Exception as e:
            print(f"[WARN] Could not read metrics.csv for {run_dir}: {e}")
    else:
        print(f"[WARN] metrics.csv not found in {run_dir}, leaving val_auc=0.0")

    best_ckpt_path = os.path.join(run_dir, "best_epoch.ckpt")
    if not os.path.exists(best_ckpt_path):
        print(
            f"[WARN] Expected best checkpoint at {best_ckpt_path} not found. "
            f"val_auc={val_auc:.4f} will be returned, best_ckpt set to null."
        )
        best_ckpt_path = None

    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(
            {
                "val_auc": val_auc,
                "best_ckpt": best_ckpt_path,
                "seed": seed,
            },
            f,
            indent=2,
        )

    return val_auc


# =============================
# Checkpoint loading helper
# =============================

def load_model_from_checkpoint(
    cfg: Dict[str, Any],
    ckpt_path: str,
    largest_tumor: Tuple[int, int, int],
    device: torch.device,
) -> ModelMultiHead:
    """
    Build imaging-only model from cfg and load weights from a Lightning checkpoint.
    We strip everything before 'backbone.' or 'heads.' and load with strict=False.
    """
    patch_size = tuple(cfg["patch_size"])
    patch_dim = int(np.prod(patch_size))
    max_num_tokens_b = compute_max_num_tokens(largest_tumor, patch_size)

    model = build_model(cfg, patch_dim, max_num_tokens_b)
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = None
        if "backbone." in k:
            idx = k.index("backbone.")
            new_key = k[idx:]
        elif "heads." in k:
            idx = k.index("heads.")
            new_key = k[idx:]
        if new_key is not None:
            new_state_dict[new_key] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading {ckpt_path}: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading {ckpt_path}: {unexpected}")

    model.eval()
    return model


# =============================
# Evaluation helper: best-epoch predictions on validation
# =============================

def evaluate_fold_best_epoch(
    run_dir: str,
    cfg: Dict[str, Any],
    data_paths: Dict[str, str],
    largest_tumor: Tuple[int, int, int],
    val_ids: List[str],
    fold_idx: int,
    slug: str,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load best_epoch.ckpt, run inference on the fold's validation set on CPU,
    and return:
      - a fold_summary dict
      - a list of per-sample prediction rows
    """
    summary: Dict[str, Any] = {
        "fold_idx": fold_idx,
        "run_dir": run_dir,
        "status": "ok",
    }
    rows: List[Dict[str, Any]] = []

    metrics_path = os.path.join(run_dir, "metrics.csv")
    summary["metrics_csv"] = metrics_path

    if os.path.exists(metrics_path):
        try:
            dfm = pd.read_csv(metrics_path)
            if "validation.metrics.auc" in dfm.columns:
                auc_col = dfm["validation.metrics.auc"].dropna()
                if len(auc_col) > 0:
                    summary["val_auc_best"] = float(auc_col.max())
                    last_row = dfm[dfm["validation.metrics.auc"].notna()].iloc[-1]
                    summary["val_auc_last"] = float(last_row["validation.metrics.auc"])
                    if "epoch" in dfm.columns:
                        best_idx = auc_col.idxmax()
                        best_epoch_val = dfm.loc[best_idx, "epoch"]
                        last_epoch_val = last_row["epoch"]
                        summary["best_epoch"] = int(best_epoch_val)
                        summary["last_epoch"] = int(last_epoch_val)
                        summary["num_epochs_trained"] = int(last_epoch_val) + 1
                else:
                    summary["status"] = "no_auc_values"
            else:
                summary["status"] = "no_auc_column"
        except Exception as e:
            print(f"[WARN] Could not parse metrics.csv for {run_dir}: {e}")
            summary["status"] = "metrics_parse_error"
    else:
        summary["status"] = "no_metrics"

    best_ckpt_path = os.path.join(run_dir, "best_epoch.ckpt")
    if not os.path.exists(best_ckpt_path):
        summary["best_ckpt"] = None
        print(f"[WARN] Cannot evaluate fold {fold_idx} in {run_dir}: best_epoch.ckpt missing.")
        return summary, rows

    summary["best_ckpt"] = best_ckpt_path

    patch_size = cfg["patch_size"]
    angle_range = (-cfg["imaging_aug_deg"], cfg["imaging_aug_deg"])
    mask_pad_threshold = cfg["mask_pad_thresh"]
    dropout_p = cfg["clinical_aug"]
    batch_size = cfg["batch_size"]
    num_workers = cfg["num_workers"]

    val_dataset = OSDataset.dataset(
        data_dir_img=data_paths["img"],
        data_dir_seg=data_paths["seg"],
        clinical_csv_path=data_paths["csv"],
        train=False,
        sample_ids=val_ids,
        patch_size=patch_size,
        largest_tumor=largest_tumor,
        angle_range=angle_range,
        mask_pad_threshold=mask_pad_threshold,
        dropout_p=dropout_p,
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=CollateDefault(),
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        worker_init_fn=worker_init_fn,
    )

    device = torch.device("cpu")
    model = load_model_from_checkpoint(
        cfg=cfg,
        ckpt_path=best_ckpt_path,
        largest_tumor=largest_tumor,
        device=device,
    )

    with torch.no_grad():
        for batch in val_dataloader:
            batch_out = model(batch)
            probs = batch_out[KEY_PROB].detach().cpu().view(-1)
            labels = batch_out[KEY_TARGET].detach().cpu().view(-1)
            sids = batch_out[KEY_SAMPLE_ID]

            if isinstance(sids, torch.Tensor):
                sids_list = [str(s) for s in sids.cpu().tolist()]
            else:
                sids_list = list(sids)

            for i in range(len(sids_list)):
                rows.append(
                    {
                        "sample_id": str(sids_list[i]),
                        "y_true": int(labels[i].item()),
                        "y_prob": float(probs[i].item()),
                        "fold_idx": fold_idx,
                        "config_slug": slug,
                        "seed": seed,
                    }
                )

    summary["n_val_samples"] = len(rows)

    return summary, rows


# =============================
# Test-set evaluation using ensemble of best-trial inner-fold models
# =============================

def evaluate_best_trial_on_test(
    best_trial: optuna.Trial,
    cfg_best: Dict[str, Any],
    runs_root: str,
    fold_idx: int,
    val_ids: List[str],
    data_paths: Dict[str, str],
    largest_tumor: Tuple[int, int, int],
    n_inner_folds: int = 5,
) -> Dict[str, Any]:
    """
    Load the n_inner_folds models saved during the best Optuna trial and evaluate
    them on the held-out test set (val_ids).

    For each inner-fold model:
      - Run inference on val_ids
      - Save per-sample predictions to test_preds_inner_fold_{k}.csv

    Ensemble step:
      - Average probabilities across all available models
      - Save to test_preds_ensemble.csv
      - Compute AUC

    Returns a summary dict (also written to test_eval_result.json).
    """
    trial_run_dir = os.path.join(runs_root, f"trial{best_trial.number:03d}_fold{fold_idx}")
    slug = config_slug(cfg_best)
    device = torch.device("cpu")

    # sample_id -> list of probabilities (one per inner-fold model)
    sample_probs: Dict[str, List[float]] = {}
    sample_labels: Dict[str, int] = {}
    fold_results: List[Dict[str, Any]] = []

    for inner_fold_idx in range(n_inner_folds):
        inner_fold_run_dir = os.path.join(trial_run_dir, f"inner_fold_{inner_fold_idx}")
        best_ckpt_path = os.path.join(inner_fold_run_dir, "best_epoch.ckpt")

        if not os.path.exists(best_ckpt_path):
            print(f"[TEST EVAL] Missing checkpoint for inner fold {inner_fold_idx}: {best_ckpt_path}")
            fold_results.append({"inner_fold_idx": inner_fold_idx, "status": "missing_ckpt"})
            continue

        model = load_model_from_checkpoint(
            cfg=cfg_best,
            ckpt_path=best_ckpt_path,
            largest_tumor=largest_tumor,
            device=device,
        )

        test_dataset = OSDataset.dataset(
            data_dir_img=data_paths["img"],
            data_dir_seg=data_paths["seg"],
            clinical_csv_path=data_paths["csv"],
            train=False,
            sample_ids=val_ids,
            patch_size=cfg_best["patch_size"],
            largest_tumor=largest_tumor,
            angle_range=(-cfg_best["imaging_aug_deg"], cfg_best["imaging_aug_deg"]),
            mask_pad_threshold=cfg_best["mask_pad_thresh"],
            dropout_p=cfg_best["clinical_aug"],
        )

        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=cfg_best["batch_size"],
            collate_fn=CollateDefault(),
            num_workers=0,
            pin_memory=False,
            shuffle=False,
        )

        fold_probs: Dict[str, float] = {}
        fold_labels: Dict[str, int] = {}

        with torch.no_grad():
            for batch in test_dataloader:
                batch_out = model(batch)
                probs = batch_out[KEY_PROB].detach().cpu().view(-1)
                labels = batch_out[KEY_TARGET].detach().cpu().view(-1)
                sids = batch_out[KEY_SAMPLE_ID]

                sids_list = (
                    [str(s) for s in sids.cpu().tolist()]
                    if isinstance(sids, torch.Tensor)
                    else list(sids)
                )

                for i, sid in enumerate(sids_list):
                    fold_probs[sid] = float(probs[i].item())
                    fold_labels[sid] = int(labels[i].item())

        # Save this fold's predictions
        fold_pred_rows = [
            {
                "sample_id": sid,
                "y_true": fold_labels[sid],
                "y_prob": fold_probs[sid],
                "inner_fold_idx": inner_fold_idx,
            }
            for sid in fold_probs
        ]
        fold_pred_df = pd.DataFrame(fold_pred_rows)
        fold_pred_path = os.path.join(
            trial_run_dir, f"test_preds_inner_fold_{inner_fold_idx}.csv"
        )
        fold_pred_df.to_csv(fold_pred_path, index=False)

        # Compute individual fold AUC
        fold_auc = None
        if fold_pred_df["y_true"].nunique() >= 2:
            fold_auc = float(roc_auc_score(fold_pred_df["y_true"], fold_pred_df["y_prob"]))
        print(
            f"[TEST EVAL] Inner fold {inner_fold_idx}: "
            f"n={len(fold_probs)}, AUC={f'{fold_auc:.4f}' if fold_auc is not None else 'N/A'}"
        )

        # Accumulate probabilities for ensemble
        for sid, prob in fold_probs.items():
            sample_probs.setdefault(sid, []).append(prob)
        sample_labels.update(fold_labels)

        fold_results.append(
            {
                "inner_fold_idx": inner_fold_idx,
                "status": "ok",
                "n_test_samples": len(fold_probs),
                "test_auc": fold_auc,
                "pred_csv": fold_pred_path,
            }
        )

    # ------------------------------------------------------------------
    # Ensemble: average probabilities across inner-fold models
    # ------------------------------------------------------------------
    ensemble_rows = [
        {
            "sample_id": sid,
            "y_true": sample_labels[sid],
            "y_prob_ensemble": float(np.mean(sample_probs[sid])),
            "n_models": len(sample_probs[sid]),
        }
        for sid in sample_probs
    ]
    df_ensemble = pd.DataFrame(ensemble_rows)
    ensemble_pred_path = os.path.join(trial_run_dir, "test_preds_ensemble.csv")
    df_ensemble.to_csv(ensemble_pred_path, index=False)

    ensemble_auc = None
    if len(df_ensemble) > 0 and df_ensemble["y_true"].nunique() >= 2:
        ensemble_auc = float(
            roc_auc_score(df_ensemble["y_true"], df_ensemble["y_prob_ensemble"])
        )
    print(
        f"[TEST EVAL] Ensemble ({sum(1 for r in fold_results if r['status'] == 'ok')} models): "
        f"n={len(df_ensemble)}, AUC={f'{ensemble_auc:.4f}' if ensemble_auc is not None else 'N/A'}"
    )

    result: Dict[str, Any] = {
        "best_trial_number": best_trial.number,
        "fold_idx": fold_idx,
        "config_slug": slug,
        "n_inner_folds_used": sum(1 for r in fold_results if r["status"] == "ok"),
        "fold_results": fold_results,
        "ensemble_pred_csv": ensemble_pred_path,
        "test_auc_ensemble": ensemble_auc,
        "n_test_samples": len(df_ensemble),
    }

    result_path = os.path.join(trial_run_dir, "test_eval_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[TEST EVAL] Results saved to {result_path}")

    return result


# =============================
# Parse arguments (similar to main_tuning.py)
# =============================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Optuna-based Imaging-only Osteosarcoma Model Tuning")
    parser.add_argument('--modality', type=str, default='T1W',
                       choices=['T1W', 'T1W_FS_C', 'T2W_FS'],
                       help='Modality of input images')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of trials for Optuna optimization')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--n_fold', type=int, default=0, help='The fold to validate on')
    parser.add_argument('--split_file', type=str, default=None,
                       help='Optional path to CSV file with predefined splits')


    
    return parser.parse_args()


# =============================
# Load or create CV splits
# =============================

def load_predefined_splits(split_file_path):
    """Load predefined splits from CSV file"""
    df = pd.read_csv(split_file_path)
    splits = []
    
    # Determine number of splits from column names
    split_columns = [col for col in df.columns if '_train' in col or '_test' in col]
    n_splits = len([col for col in split_columns if '_train' in col])
    
    print(f"Loaded {n_splits} splits from {split_file_path}")
    
    for i in range(n_splits):
        train_col = f'{i}_train'
        test_col = f'{i}_test'
        
        if train_col in df.columns and test_col in df.columns:
            train_patients = df[train_col].dropna().tolist()
            test_patients = df[test_col].dropna().tolist()
            
            splits.append({
                'train': train_patients,
                'test': test_patients
            })
            print(f"Split {i}: {len(train_patients)} train, {len(test_patients)} test patients")
        else:
            print(f"Warning: Columns {train_col} or {test_col} not found in split file")
    
    return splits


# =============================
# MAIN: Optuna-based HPO with CV
# =============================

def main():
    torch.set_float32_matmul_precision('high')
    args = parse_arguments()

    
    GLOBAL_SEED = args.random_seed
    seed_everything(GLOBAL_SEED)

    # Data paths
    data_dir = f'/projects/prjs1779/Osteosarcoma/exp_data/{args.modality}/v1/'
    data_dir_img = os.path.join(data_dir, "input", "img")
    data_dir_seg = os.path.join(data_dir, "input", "seg")
    clinical_csv_path = os.path.join(data_dir, "clinical_features_with_Huvos.csv")
    args.split_file = os.path.join(data_dir, "patient_splits.csv")
    data_paths = {"img": data_dir_img, "seg": data_dir_seg, "csv": clinical_csv_path}

    # Largest tumor dimensions (adjust based on your data)
    largest_tumor = (32, 286, 277)

    # Base configuration
    base_cfg = {
        "patch_size": (16, 64, 64),
        "emb_dim": 64,
        "lr": 1e-4,
        "wd": 1e-3,
        "batch_size": 8,
        "num_workers": 10,
        "depth_a": 2, "heads_a": 2,
        "depth_b": 2, "heads_b": 4,
        "depth_cross_attn": 2, "heads_cross": 2,
        "mlp_layers": "single",
        "mask_pad_thresh": 0.7,
        "clinical_aug": 0.0,
        "imaging_aug_deg": 0,
    }

    SCRATCH_DIR_NAME = "/scratch-shared/xwan1"
    runs_root = os.path.join(SCRATCH_DIR_NAME, RUNS_DIR_NAME)
    ensure_dir(runs_root)
    runs_root = os.path.join(runs_root, f"{args.modality}", f"{args.n_fold}")
    ensure_dir(runs_root)

    

    # Load or create CV splits
    cv_splits = load_predefined_splits(
        split_file_path=args.split_file,
    )
    print(f"Loaded {len(cv_splits)} CV folds")

    # Create Optuna study
    db_path = os.path.join(runs_root, 'optuna_study.db')
    study = optuna.create_study(
        direction="maximize",
        pruner=SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0),
        study_name=OPTUNA_STUDY_NAME,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )

    # Stable directory that always holds the latest best-trial test predictions.
    # Overwritten whenever a new best trial is found during optimization.
    current_best_preds_dir = os.path.join(runs_root, "current_best_test_preds")

    # Mutable tracker so the objective closure can update it across calls.
    best_tracker: Dict[str, Any] = {
        "score": -1.0,
        "trial_number": -1,
        "test_auc_ensemble": None,
    }

    # Define inner CV function
    def run_inner_cv(cfg: Dict[str, Any], 
                     train_ids: List[str], 
                     val_ids: List[str],
                     fold_idx: int,
                     trial: optuna.Trial,
                     run_dir_base: str) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Run 5-fold inner cross-validation on the training data.
        Similar to CrossValidationFramework, splits by sample IDs stratified by label.
        Returns mean validation AUC from inner folds and prediction rows.
        """
        # Load full dataset to get labels for all samples
        full_dataset = OSDataset.dataset(
            data_dir_img=data_paths["img"],
            data_dir_seg=data_paths["seg"],
            clinical_csv_path=data_paths["csv"],
            train=False,
            sample_ids=train_ids,
            patch_size=cfg["patch_size"],
            largest_tumor=largest_tumor,
            angle_range=(0.0, 0.0),
            mask_pad_threshold=0.8,
            dropout_p=0.0,
        )
        
        # Get labels for all training samples
        sample_id_to_label = {}
        for sample in full_dataset:
            sid = sample[KEY_SAMPLE_ID]
            label = sample[KEY_TARGET]
            sample_id_to_label[sid] = label
        
        # Get labels for stratification
        train_labels = [sample_id_to_label.get(sid, 0) for sid in train_ids]
        
        # Create stratified inner folds
        inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED + fold_idx)
        
        inner_val_aucs = []
        
        for inner_fold_idx, (train_split_idx, val_split_idx) in enumerate(            
            inner_skf.split(train_ids, train_labels)):

            print(f"Inner Fold {inner_fold_idx + 1}/5")
            
            # Get sample IDs for this inner fold
            inner_train_ids = [train_ids[i] for i in train_split_idx]
            inner_val_ids = [train_ids[i] for i in val_split_idx]
            
            inner_fold_run_dir = os.path.join(
                run_dir_base,
                f"inner_fold_{inner_fold_idx}"
            )
            ensure_dir(inner_fold_run_dir)
            
            print(f"[OPTUNA] Trial {trial.number}, Outer Fold {fold_idx}, Inner Fold {inner_fold_idx}")
            print(f"  Inner train: {len(inner_train_ids)}, Inner val: {len(inner_val_ids)}")
            
            # Train model on inner train set
            inner_val_auc = train_one_trial(
                run_dir=inner_fold_run_dir,
                cfg=cfg,
                data_paths=data_paths,
                largest_tumor=largest_tumor,
                train_ids=inner_train_ids, 
                val_ids=inner_val_ids, 
                seed=GLOBAL_SEED + fold_idx * 100 + inner_fold_idx,
            )
            
            inner_val_aucs.append(inner_val_auc)
            print(f"  Inner fold {inner_fold_idx} val_auc: {inner_val_auc:.4f}")
        
        mean_inner_auc = float(np.mean(inner_val_aucs)) if len(inner_val_aucs) > 0 else 0.0
        print(f"[OPTUNA] Trial {trial.number}, Outer Fold {fold_idx}: mean_inner_auc={mean_inner_auc:.4f}")
        
        return mean_inner_auc
    
    # Define objective function
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        cfg = suggest_hyperparameters(trial, base_cfg)
        slug = config_slug(cfg)

        print(f"\n[OPTUNA] Trial {trial.number} - Config slug: {slug}")
        report = {k: cfg.get(k) for k in REPORT_KEYS if k in cfg}
        print(f"[OPTUNA] Trial {trial.number} - CONFIG: {report}")

        val_aucs: List[float] = []
        fold_summaries: List[Dict[str, Any]] = []

        # Run outer CV folds
        for fold_idx, split in enumerate(cv_splits):
            
            if fold_idx != args.n_fold:
                continue
            # Remove 'OS_000085_02' from splits if present
            split['train'] = [sid for sid in split['train'] if sid != 'OS_000085_02']
            split['test'] = [sid for sid in split['test'] if sid != 'OS_000085_02'] 
            
            train_ids = split['train']
            val_ids = split['test']
            fold_run_dir = os.path.join(
                runs_root,
                f"trial{trial.number:03d}_fold{fold_idx}"
            )
            
            print(f"\n[OPTUNA] Trial {trial.number}, Outer Fold {fold_idx}/{len(cv_splits)}")
            print(f"  Train IDs: {len(train_ids)}, Test IDs: {len(val_ids)}")
            
            # Run 5-fold inner CV on training data
            mean_inner_auc = run_inner_cv(
                cfg=cfg,
                train_ids=train_ids, 
                val_ids=val_ids, 
                fold_idx=fold_idx,
                trial=trial,
                run_dir_base=fold_run_dir,
            )
            
            print(f"[OPTUNA] Trial {trial.number}, Outer Fold {fold_idx}: mean_inner_auc={mean_inner_auc:.4f}")
            val_aucs.append(mean_inner_auc)
            # all_pred_rows.append(inner_pred_rows)

        # Compute aggregate metrics
        AUC_dev: Optional[float] = None
        n_dev_samples: int = 0

        # if len(all_pred_rows) > 0:
        #     df_pred = pd.DataFrame(all_pred_rows)
        #     n_dev_samples = len(df_pred)

        #     if df_pred["y_true"].nunique() >= 2:
        #         AUC_dev = float(roc_auc_score(df_pred["y_true"], df_pred["y_prob"]))
        #     else:
        #         AUC_dev = None

        #     if AUC_dev is not None:
        #         score = AUC_dev
        #     else:
        #         score = 0.0
        # else:
        #     score = 0.0

        mean_val_auc = float(np.mean(val_aucs)) if len(val_aucs) > 0 else 0.0
        AUC_dev = mean_val_auc
        score = AUC_dev 

        # Save trial results
        # if len(all_pred_rows) > 0:
        #     df_pred = pd.DataFrame(all_pred_rows)
        #     pred_csv_path = os.path.join(runs_root, f"trial{trial.number:03d}_{slug}_predictions.csv")
        #     df_pred.to_csv(pred_csv_path, index=False)
        # else:
        #     pred_csv_path = None

        aggregate_path = os.path.join(runs_root, f"trial{trial.number:03d}_{slug}_aggregate.json")
        aggregate_payload: Dict[str, Any] = {
            "trial_number": trial.number,
            "config_slug": slug,
            "cfg": cfg,
            "seed": GLOBAL_SEED,
            "fold_val_aucs": val_aucs,
            "mean_val_auc": mean_val_auc,
            "folds": fold_summaries,
            "dev_metrics": {
                "AUC_dev": AUC_dev,
                "n_dev_samples": n_dev_samples,
            },
            "objective": {
                "formula": "Mean validation AUC",
                "AUC_dev": AUC_dev,
                # "score": score,
            },
            # "predictions_csv": pred_csv_path,
        }

        with open(aggregate_path, "w") as f:
            json.dump(aggregate_payload, f, indent=2)

        print(
            f"[OPTUNA] Trial {trial.number}: mean_val_auc={mean_val_auc:.4f}, "
            f"AUC_dev={AUC_dev}"
        )

        # Set trial attributes for logging
        # trial.set_user_attr("config_slug", slug)
        trial.set_user_attr("AUC_dev", AUC_dev)
        # trial.set_user_attr("mean_val_auc", mean_val_auc)

        # ------------------------------------------------------------------
        # If this trial is a new best, evaluate immediately on the test set
        # and update the stable "current_best_test_preds" directory.
        # ------------------------------------------------------------------
        if score > best_tracker["score"]:
            best_tracker["score"] = score
            best_tracker["trial_number"] = trial.number

            print(
                f"\n[BEST TRIAL UPDATE] Trial {trial.number} is new best "
                f"(AUC_dev={score:.4f}). Running test-set evaluation ..."
            )

            test_eval = evaluate_best_trial_on_test(
                best_trial=trial,
                cfg_best=cfg,
                runs_root=runs_root,
                fold_idx=args.n_fold,
                val_ids=val_ids,
                data_paths=data_paths,
                largest_tumor=largest_tumor,
                n_inner_folds=5,
            )
            best_tracker["test_auc_ensemble"] = test_eval["test_auc_ensemble"]

            # Copy predictions to the stable "current best" location,
            # overwriting whatever was there from a previous best trial.
            ensure_dir(current_best_preds_dir)
            trial_run_dir_best = os.path.join(
                runs_root, f"trial{trial.number:03d}_fold{args.n_fold}"
            )

            ensemble_src = os.path.join(trial_run_dir_best, "test_preds_ensemble.csv")
            if os.path.exists(ensemble_src):
                shutil.copy2(
                    ensemble_src,
                    os.path.join(current_best_preds_dir, "test_preds_ensemble.csv"),
                )

            for k in range(5):
                fold_src = os.path.join(
                    trial_run_dir_best, f"test_preds_inner_fold_{k}.csv"
                )
                if os.path.exists(fold_src):
                    shutil.copy2(
                        fold_src,
                        os.path.join(
                            current_best_preds_dir, f"test_preds_inner_fold_{k}.csv"
                        ),
                    )

            with open(
                os.path.join(current_best_preds_dir, "best_trial_info.json"), "w"
            ) as f:
                json.dump(
                    {
                        "trial_number": trial.number,
                        "AUC_dev": score,
                        "test_auc_ensemble": test_eval["test_auc_ensemble"],
                        "n_test_samples": test_eval["n_test_samples"],
                    },
                    f,
                    indent=2,
                )

            print(
                f"[BEST TRIAL UPDATE] Test ensemble AUC={test_eval['test_auc_ensemble']}, "
                f"predictions saved to {current_best_preds_dir}"
            )

        return score

    # Run optimization
    print(f"\n[OPTUNA] Starting optimization with {args.n_trials} trials")
    study.optimize(objective, n_trials=args.n_trials)

    # Print results
    print(f"\n[OPTUNA] Optimization finished!")
    print(f"[OPTUNA] Best trial: {study.best_trial.number}")
    print(f"[OPTUNA] Best value (score): {study.best_trial.value:.4f}")
    print(f"[OPTUNA] Best hyperparameters: {study.best_trial.params}")

    # Save best trial info (test evaluation was already run during optimization)
    best_trial = study.best_trial
    best_info_path = os.path.join(runs_root, "best_trial.json")
    with open(best_info_path, "w") as f:
        json.dump({
            "trial_number": best_trial.number,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "best_attributes": best_trial.user_attrs,
            "test_auc_ensemble": best_tracker["test_auc_ensemble"],
            "current_best_preds_dir": current_best_preds_dir,
        }, f, indent=2)
    print(
        f"[TEST EVAL] Final best trial {best_trial.number}: "
        f"AUC_dev={best_trial.value:.4f}, "
        f"test_auc_ensemble={best_tracker['test_auc_ensemble']}"
    )

    print(f"[OPTUNA] Results saved to {runs_root}")


if __name__ == "__main__":
    main()
