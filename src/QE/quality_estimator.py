import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Config
# =========================================================

@dataclass
class TrainConfig:
    txt_path: str = "dataset/clothing/text_feat.npy"
    img_path: str = "dataset/clothing/image_feat.npy"
    save_dir: str = os.path.join(
        os.path.dirname(__file__),
        "quality_estimator_ckpt",
        "clothing",
    )

    seed: int = 42
    valid_item_start: int = 1   # RecBole padding item 0 제외

    # split
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # training
    batch_size: int = 512
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 512
    state_emb_dim: int = 64
    patience: int = 8
    num_workers: int = 0

    # train-time corruption probability
    p_clean: float = 0.4
    p_noisy: float = 0.3
    p_missing: float = 0.3

    # noisy type
    noisy_types: Tuple[str, ...] = ("gaussian", "dropout")
    gaussian_std_choices: Tuple[float, ...] = (0.3, 0.5, 0.7)
    dropout_prob_choices: Tuple[float, ...] = (0.3, 0.5, 0.7)

    # fixed val/test corruption config
    eval_p_clean: float = 1 / 3
    eval_p_noisy: float = 1 / 3
    eval_p_missing: float = 1 / 3
    eval_repeats_per_item: int = 1

    # monotone ranking loss (quality_estimator pretraining only; run.py sets this to 0)
    lambda_mon: float = 1.0
    margin_mon: float = 0.1

    # severity supervision loss: MSE(severity, target) where clean->0, noisy->0.5, missing->1
    lambda_sev: float = 1.0

    label_smoothing: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


STATE_MAP = {
    "clean":   0,
    "noisy":   1,
    "missing": 2,
}
ID2STATE = {v: k for k, v in STATE_MAP.items()}


# =========================================================
# Utils
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def stratified_split_dummy(indices: np.ndarray, val_ratio: float, test_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = indices.copy()
    rng.shuffle(idx)

    n = len(idx)
    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))

    train_idx = idx[:n - n_val - n_test]
    val_idx   = idx[n - n_val - n_test:n - n_test]
    test_idx  = idx[n - n_test:]
    return train_idx, val_idx, test_idx


def compute_class_weights_from_probs(p_clean: float, p_noisy: float, p_missing: float):
    probs   = np.array([p_clean, p_noisy, p_missing], dtype=np.float32)
    probs   = probs / probs.sum()
    weights = 1.0 / probs
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_confmat(preds: np.ndarray, targets: np.ndarray, num_classes: int = 3) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(preds, targets):
        cm[t, p] += 1
    return cm


def macro_f1_from_confmat(confmat: np.ndarray) -> float:
    f1s = []
    for c in range(confmat.shape[0]):
        tp    = confmat[c, c]
        fp    = confmat[:, c].sum() - tp
        fn    = confmat[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def print_confmat(cm: np.ndarray, title: str = ""):
    names = ["clean", "noisy", "missing"]
    if title:
        print(title)
    print("rows=true, cols=pred")
    print("            " + " ".join([f"{n:>10}" for n in names]))
    for i, row in enumerate(cm):
        print(f"{names[i]:>10} " + " ".join([f"{v:10d}" for v in row]))


def tensorize(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x)


# =========================================================
# Corruption functions
# =========================================================

def apply_gaussian_noise(x: torch.Tensor, std: float, g: torch.Generator = None) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype) * std
    return x + noise


def apply_dropout_noise(x: torch.Tensor, drop_prob: float, g: torch.Generator = None) -> torch.Tensor:
    rand = torch.rand(x.shape, generator=g, device=x.device, dtype=x.dtype)
    mask = (rand > drop_prob).to(x.dtype)
    return x * mask


def corrupt_embedding(x: torch.Tensor, state_id: int, cfg: TrainConfig, g: torch.Generator = None):
    x = F.normalize(x, dim=-1)  # unit sphere 위에서 corruption 적용

    if state_id == STATE_MAP["clean"]:
        return x

    if state_id == STATE_MAP["missing"]:
        return torch.zeros_like(x)

    # noisy
    noisy_type = random.choice(cfg.noisy_types)
    if noisy_type == "gaussian":
        std = random.choice(cfg.gaussian_std_choices)
        return apply_gaussian_noise(x, std=std, g=g)
    elif noisy_type == "dropout":
        drop_prob = random.choice(cfg.dropout_prob_choices)
        return apply_dropout_noise(x, drop_prob=drop_prob, g=g)
    else:
        raise ValueError(f"Unknown noisy type: {noisy_type}")


# =========================================================
# Dataset
# =========================================================

class OnTheFlyCorruptionDataset(Dataset):
    """Train dataset: __getitem__ 호출마다 랜덤 corruption 적용."""

    def __init__(self, clean_x: torch.Tensor, cfg: TrainConfig):
        self.clean_x = clean_x.float()
        self.cfg     = cfg

        probs = np.array([cfg.p_clean, cfg.p_noisy, cfg.p_missing], dtype=np.float64)
        self.class_probs = probs / probs.sum()

    def __len__(self):
        return self.clean_x.size(0)

    def __getitem__(self, idx):
        x   = self.clean_x[idx]
        y   = int(np.random.choice([0, 1, 2], p=self.class_probs))
        x_c = corrupt_embedding(x, y, self.cfg)
        return x_c, torch.tensor(y, dtype=torch.long)


class TripletCorruptionDataset(Dataset):
    """Train dataset for L_mon: returns (x_clean, x_noisy, x_missing) per item."""

    def __init__(self, clean_x: torch.Tensor, cfg: TrainConfig):
        self.clean_x = clean_x.float()
        self.cfg     = cfg

    def __len__(self):
        return self.clean_x.size(0)

    def __getitem__(self, idx):
        x = self.clean_x[idx]
        return (
            x.clone(),
            corrupt_embedding(x, STATE_MAP["noisy"],   self.cfg),
            corrupt_embedding(x, STATE_MAP["missing"], self.cfg),
        )


class FixedCorruptionDataset(Dataset):
    """Val/Test dataset: 고정 corruption."""

    def __init__(self, x_corr: torch.Tensor, y: torch.Tensor):
        self.x_corr = x_corr.float()
        self.y      = y.long()

    def __len__(self):
        return self.x_corr.size(0)

    def __getitem__(self, idx):
        return self.x_corr[idx], self.y[idx]


def build_fixed_eval_set(clean_x: torch.Tensor, cfg: TrainConfig, seed: int):
    g   = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    probs = np.array([cfg.eval_p_clean, cfg.eval_p_noisy, cfg.eval_p_missing], dtype=np.float64)
    probs = probs / probs.sum()

    xs, ys = [], []
    for i in range(clean_x.size(0)):
        x = clean_x[i]
        for _ in range(cfg.eval_repeats_per_item):
            y   = int(rng.choice([0, 1, 2], p=probs))
            x_c = corrupt_embedding(x, y, cfg, g=g)
            xs.append(x_c.unsqueeze(0))
            ys.append(y)

    return torch.cat(xs, dim=0), torch.tensor(ys, dtype=torch.long)


# =========================================================
# Model
# =========================================================

class QualityEstimator(nn.Module):
    """Per-modality quality estimator.

    Input : x  [B, d]
    Output: dict
        prob        [B, 3]   softmax(logits) — [p_clean, p_noisy, p_missing]
        p_clean     [B, 1]
        p_noisy     [B, 1]
        p_missing   [B, 1]
        state_emb   [B, d_state]
        severity    [B, 1]   learnable monotone sigmoid
        reliability [B, 1]   = 1 - severity
        uncertainty [B, 1]   normalized posterior entropy
    """

    def __init__(self, input_dim: int, hidden_dim: int, state_emb_dim: int):
        super().__init__()

        # MLP classifier -> [B, 3]
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # State embedding: small MLP on top of prob
        self.state_proj = nn.Sequential(
            nn.Linear(3, state_emb_dim),
            nn.ReLU(),
        )

        # Learnable monotone severity parameters
        # alpha_noisy   = softplus(a_noisy)
        # alpha_missing = alpha_noisy + softplus(delta)  => alpha_missing >= alpha_noisy >= 0
        self.a_noisy = nn.Parameter(torch.zeros(1))
        self.delta   = nn.Parameter(torch.zeros(1))
        self.b       = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> dict:
        x = F.normalize(x, dim=-1)              # inference 시에도 unit sphere로 통일
        logits = self.classifier(x)             # [B, 3]
        prob   = F.softmax(logits, dim=-1)      # [B, 3]

        p_clean   = prob[:, 0:1]
        p_noisy   = prob[:, 1:2]
        p_missing = prob[:, 2:3]

        state_emb = self.state_proj(prob)       # [B, d_state]

        alpha_noisy   = F.softplus(self.a_noisy)
        alpha_missing = alpha_noisy + F.softplus(self.delta)

        severity    = torch.sigmoid(
            alpha_noisy * p_noisy + alpha_missing * p_missing + self.b
        )                                       # [B, 1]
        reliability = 1.0 - severity            # [B, 1]
        uncertainty = -(
            prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / np.log(3.0)

        return {
            "logits":      logits,
            "prob":        prob,
            "p_clean":     p_clean,
            "p_noisy":     p_noisy,
            "p_missing":   p_missing,
            "state_emb":   state_emb,
            "severity":    severity,
            "reliability": reliability,
            "uncertainty": uncertainty,
        }

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["prob"]          # [B, 3]

    @torch.no_grad()
    def uncertainty_score(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["uncertainty"].squeeze(-1)  # [B]


# =========================================================
# Data loading & split
# =========================================================

def _load_emb_file(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path)).float()
    return tensorize(torch.load(path, map_location="cpu", weights_only=True)).float()


def load_clean_embeddings(cfg: TrainConfig):
    txt = _load_emb_file(cfg.txt_path)
    img = _load_emb_file(cfg.img_path)

    txt = txt[cfg.valid_item_start:]
    img = img[cfg.valid_item_start:]

    assert txt.size(0) == img.size(0), "text/image item 수가 다릅니다."
    return txt, img


def build_modality_loaders(clean_x: torch.Tensor, cfg: TrainConfig, modality_seed: int):
    n = clean_x.size(0)
    train_idx, val_idx, test_idx = stratified_split_dummy(
        np.arange(n), cfg.val_ratio, cfg.test_ratio, seed=modality_seed
    )

    train_ds = OnTheFlyCorruptionDataset(clean_x[train_idx], cfg)

    x_val,  y_val  = build_fixed_eval_set(clean_x[val_idx],  cfg, seed=modality_seed + 100)
    x_test, y_test = build_fixed_eval_set(clean_x[test_idx], cfg, seed=modality_seed + 200)

    val_ds  = FixedCorruptionDataset(x_val,  y_val)
    test_ds = FixedCorruptionDataset(x_test, y_test)

    triplet_ds = TripletCorruptionDataset(clean_x[train_idx], cfg)

    kwargs = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    train_loader   = DataLoader(train_ds,   shuffle=True,  **kwargs)
    triplet_loader = DataLoader(triplet_ds, shuffle=True,  **kwargs)
    val_loader     = DataLoader(val_ds,     shuffle=False, **kwargs)
    test_loader    = DataLoader(test_ds,    shuffle=False, **kwargs)

    return train_loader, triplet_loader, val_loader, test_loader, clean_x[train_idx], clean_x[val_idx], clean_x[test_idx]


# =========================================================
# Train / Eval
# =========================================================

_SEV_TARGET = torch.tensor([0.0, 0.5, 1.0])  # clean, noisy, missing


def run_epoch(model: QualityEstimator, loader, optimizer, criterion, device, train: bool = True,
              triplet_loader=None, lambda_mon: float = 0.0, margin_mon: float = 0.1,
              lambda_sev: float = 0.0):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    triplet_iter = iter(triplet_loader) if (train and lambda_mon > 0 and triplet_loader is not None) else None

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            out    = model(x)
            logits = out["logits"]              # [B, 3]
            loss   = criterion(logits, y)

            if lambda_sev > 0:
                sev_target = _SEV_TARGET.to(device)[y]  # [B]
                l_sev = F.mse_loss(out["severity"].squeeze(-1), sev_target)
                loss  = loss + lambda_sev * l_sev

            if triplet_iter is not None:
                try:
                    xc, xn, xm = next(triplet_iter)
                except StopIteration:
                    triplet_iter = iter(triplet_loader)
                    xc, xn, xm  = next(triplet_iter)
                xc, xn, xm = xc.to(device), xn.to(device), xm.to(device)
                u_c = model(xc)["severity"]     # [B, 1]
                u_n = model(xn)["severity"]
                u_m = model(xm)["severity"]
                l_mon = (
                    F.relu(margin_mon + u_c - u_n) +
                    F.relu(margin_mon + u_n - u_m)
                ).mean()
                loss = loss + lambda_mon * l_mon

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(dim=-1).detach().cpu())
        all_targets.append(y.detach().cpu())

    all_preds   = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    cm       = build_confmat(all_preds, all_targets)
    acc      = float((all_preds == all_targets).mean())
    macro_f1 = macro_f1_from_confmat(cm)

    return {
        "loss":     total_loss / len(loader.dataset),
        "acc":      acc,
        "macro_f1": macro_f1,
        "confmat":  cm,
        "preds":    all_preds,
        "targets":  all_targets,
    }


def save_checkpoint(path: str, model: QualityEstimator, cfg: TrainConfig, extra: Dict):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config":           asdict(cfg),
            "extra":            extra,
            "state_map":        STATE_MAP,
        },
        path,
    )


def summarize_uncertainty_scores(model: QualityEstimator, clean_x: torch.Tensor,
                                  cfg: TrainConfig, seed: int, name: str):
    x_eval, y_eval = build_fixed_eval_set(clean_x, cfg, seed=seed)
    model.eval()

    with torch.no_grad():
        u = model.uncertainty_score(x_eval.to(cfg.device)).cpu()

    u    = u.numpy()
    y_np = y_eval.numpy()

    print(f"\n[{name.upper()} uncertainty score stats]")
    print(f"  formula: learnable monotone severity (sigmoid(alpha_n*p_n + alpha_m*p_m + b))")
    for cls_id, cls_name in ID2STATE.items():
        vals = u[y_np == cls_id]
        print(
            f"  [{cls_name:<8}] n={len(vals):6d} "
            f"mean={vals.mean():.4f} std={vals.std():.4f} "
            f"p10={np.quantile(vals, 0.10):.4f} "
            f"p25={np.quantile(vals, 0.25):.4f} "
            f"p50={np.quantile(vals, 0.50):.4f} "
            f"p75={np.quantile(vals, 0.75):.4f} "
            f"p90={np.quantile(vals, 0.90):.4f}"
        )


# =========================================================
# Main training function
# =========================================================

def train_one_estimator(modality_name: str, clean_x: torch.Tensor, cfg: TrainConfig, seed_offset: int):
    print("\n" + "=" * 90)
    print(f"[TRAIN {modality_name.upper()} QUALITY ESTIMATOR]")
    print("=" * 90)

    train_loader, triplet_loader, val_loader, test_loader, train_clean, val_clean, test_clean = build_modality_loaders(
        clean_x, cfg, modality_seed=cfg.seed + seed_offset
    )

    input_dim = clean_x.size(1)
    model = QualityEstimator(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        state_emb_dim=cfg.state_emb_dim,
    ).to(cfg.device)

    class_weights = compute_class_weights_from_probs(cfg.p_clean, cfg.p_noisy, cfg.p_missing).to(cfg.device)
    print(f"[{modality_name}] class weights = {class_weights.detach().cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1 = -1.0
    best_path   = os.path.join(cfg.save_dir, f"{modality_name}_quality_estimator_best.pt")
    history     = []
    no_improve  = 0

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, criterion, cfg.device, train=True,
                       triplet_loader=triplet_loader, lambda_mon=cfg.lambda_mon, margin_mon=cfg.margin_mon,
                       lambda_sev=cfg.lambda_sev)
        va = run_epoch(model, val_loader, optimizer=None, criterion=criterion, device=cfg.device, train=False)

        scheduler.step(va["macro_f1"])

        log = {
            "epoch":           epoch,
            "train_loss":      tr["loss"],
            "train_acc":       tr["acc"],
            "train_macro_f1":  tr["macro_f1"],
            "val_loss":        va["loss"],
            "val_acc":         va["acc"],
            "val_macro_f1":    va["macro_f1"],
            "lr":              optimizer.param_groups[0]["lr"],
        }
        history.append(log)

        print(
            f"[{modality_name}] epoch {epoch:03d} | "
            f"train loss {tr['loss']:.4f} acc {tr['acc']:.4f} f1 {tr['macro_f1']:.4f} | "
            f"val loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['macro_f1']:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )

        if va["macro_f1"] > best_val_f1:
            best_val_f1 = va["macro_f1"]
            no_improve  = 0
            save_checkpoint(
                best_path, model, cfg,
                extra={
                    "modality":           modality_name,
                    "input_dim":          input_dim,
                    "best_val_macro_f1":  best_val_f1,
                    "history":            history,
                }
            )
            print(f"[{modality_name}] best checkpoint saved -> {best_path}")
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            print(f"[{modality_name}] early stopping triggered.")
            break

    # 베스트 체크포인트 로드 후 test 평가
    ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    te = run_epoch(model, test_loader, optimizer=None, criterion=criterion, device=cfg.device, train=False)

    print("\n" + "-" * 90)
    print(f"[{modality_name.upper()} TEST RESULT]")
    print(f"loss={te['loss']:.4f}  acc={te['acc']:.4f}  macro_f1={te['macro_f1']:.4f}")
    print_confmat(te["confmat"], title=f"[{modality_name}] confusion matrix")
    print("-" * 90)

    summarize_uncertainty_scores(
        model, clean_x=test_clean, cfg=cfg,
        seed=cfg.seed + seed_offset + 999, name=modality_name
    )

    return {
        "best_val_macro_f1": best_val_f1,
        "test_loss":         te["loss"],
        "test_acc":          te["acc"],
        "test_macro_f1":     te["macro_f1"],
        "test_confmat":      te["confmat"].tolist(),
        "best_ckpt_path":    best_path,
    }


# =========================================================
# Real-data evaluation
# =========================================================

def evaluate_on_real_data(ckpt_path: str, x_corr: torch.Tensor, y_true: torch.Tensor,
                           modality_name: str, device: str):
    """Evaluate a saved checkpoint on real corrupted data (mixed40_mm_emb.pt style)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]

    model = QualityEstimator(
        input_dim=ckpt["extra"]["input_dim"],
        hidden_dim=cfg_dict["hidden_dim"],
        state_emb_dim=cfg_dict["state_emb_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_corr = x_corr.float().to(device)
    y_true  = y_true.long()

    with torch.no_grad():
        out      = model(x_corr)
        preds    = out["logits"].argmax(dim=-1).cpu().numpy()
        severity = out["severity"].squeeze(-1).cpu().numpy()

    y_np = y_true.numpy()
    cm   = build_confmat(preds, y_np)
    acc  = float((preds == y_np).mean())
    f1   = macro_f1_from_confmat(cm)

    print(f"\n{'=' * 90}")
    print(f"[REAL-DATA EVAL — {modality_name.upper()}]  (mixed40_mm_emb.pt)")
    print(f"  acc={acc:.4f}  macro_f1={f1:.4f}")
    print_confmat(cm, title=f"  [{modality_name}] confusion matrix")

    print(f"\n  [{modality_name}] severity stats per true class:")
    for cls_id, cls_name in ID2STATE.items():
        vals = severity[y_np == cls_id]
        if len(vals) == 0:
            continue
        print(
            f"    [{cls_name:<8}] n={len(vals):6d} "
            f"mean={vals.mean():.4f} std={vals.std():.4f} "
            f"p25={np.quantile(vals, 0.25):.4f} "
            f"p50={np.quantile(vals, 0.50):.4f} "
            f"p75={np.quantile(vals, 0.75):.4f}"
        )
    print(f"{'=' * 90}")

    return {"acc": acc, "macro_f1": f1, "confmat": cm.tolist()}


# =========================================================
# Main
# =========================================================

def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)

    print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))

    txt_clean, img_clean = load_clean_embeddings(cfg)
    print(f"[TEXT CLEAN]  shape={tuple(txt_clean.shape)}")
    print(f"[IMAGE CLEAN] shape={tuple(img_clean.shape)}")

    text_result  = train_one_estimator("text",  txt_clean, cfg, seed_offset=10)
    image_result = train_one_estimator("image", img_clean, cfg, seed_offset=20)

    summary = {
        "config":       asdict(cfg),
        "text_result":  text_result,
        "image_result": image_result,
    }

    # ── real-data evaluation on mixed40_mm_emb.pt ──────────────────────────
    real_path = "data/clothing/missing/mixed40_mm_emb.pt"
    if os.path.exists(real_path):
        real = torch.load(real_path, map_location="cpu", weights_only=False)
        real_results = {}
        for mod, x_key, y_key in [
            ("text",  "txt_emb_corrupted", "text_state"),
            ("image", "img_emb_corrupted", "image_state"),
        ]:
            ckpt_path = os.path.join(cfg.save_dir, f"{mod}_quality_estimator_best.pt")
            real_results[mod] = evaluate_on_real_data(
                ckpt_path=ckpt_path,
                x_corr=real[x_key][cfg.valid_item_start:],
                y_true=real[y_key][cfg.valid_item_start:],
                modality_name=mod,
                device=cfg.device,
            )
        summary["real_data_results"] = real_results
    else:
        print(f"\n[WARN] real data not found: {real_path}")
    # ────────────────────────────────────────────────────────────────────────

    summary_path = os.path.join(cfg.save_dir, "train_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
