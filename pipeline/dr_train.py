"""
Training helpers that IMPORT RETFound's real components (no reimplementation):
  * models_vit.RETFound_mae          -- ViT-L backbone w/ global_pool
  * util.pos_embed.interpolate_pos_embed
  * util.lr_decay.param_groups_lrd   -- layer-wise LR decay
  * util.datasets.build_dataset      -- ImageFolder + RETFound transforms
  * engine_finetune.train_one_epoch  -- the recipe's train loop
  * util.misc.NativeScalerWithGradNormCount

The notebook orchestrates these; keeping them here lets us smoke-test without
running the notebook and keeps the join/recipe auditable.
"""
import os
# Reduce CUDA fragmentation (set before the caching allocator initialises, i.e. before
# the first CUDA op). Honoured as long as dr_train is imported before any model.to(cuda).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import types
import argparse
import numpy as np
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True   # tolerate the one truncated source JPEG

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RETFOUND_REPO = os.environ.get("RETFOUND_REPO", os.path.join(PROJECT_ROOT, "RETFound_repo"))


def _ensure_repo_on_path():
    if RETFOUND_REPO not in sys.path:
        sys.path.insert(0, RETFOUND_REPO)


def make_args(cfg):
    """Build the argparse.Namespace that RETFound's functions expect."""
    a = argparse.Namespace()
    # data / model
    a.data_path = cfg["data_path"]
    a.nb_classes = cfg["nb_classes"]
    a.input_size = cfg["input_size"]
    a.model = cfg.get("model", "RETFound_mae")          # "RETFound_mae" | "RETFound_dinov2"
    a.model_arch = cfg.get("model_arch", "retfound_mae")
    a.finetune = cfg["finetune_id"]
    a.drop_path = cfg["drop_path"]
    a.global_pool = True
    a.adaptation = cfg.get("adaptation", "finetune")
    # optim
    a.batch_size = cfg["batch_size"]
    a.accum_iter = cfg["accum_iter"]
    a.epochs = cfg["epochs"]
    a.warmup_epochs = cfg["warmup_epochs"]
    a.blr = cfg["blr"]
    a.lr = None
    a.layer_decay = cfg["layer_decay"]
    a.weight_decay = cfg["weight_decay"]
    a.min_lr = cfg["min_lr"]
    a.clip_grad = cfg.get("clip_grad", None)
    # augmentation (RETFound defaults)
    a.color_jitter = None
    a.aa = cfg.get("aa", "rand-m9-mstd0.5-inc1")
    a.smoothing = cfg.get("smoothing", 0.1)
    a.reprob = cfg.get("reprob", 0.25)
    a.remode = "pixel"
    a.recount = 1
    a.resplit = False
    # mixup OFF (RETFound_mae finetune default) -> lets us use weighted CE
    a.mixup = 0.0; a.cutmix = 0.0; a.cutmix_minmax = None
    a.mixup_prob = 1.0; a.mixup_switch_prob = 0.5; a.mixup_mode = "batch"
    # runtime
    a.device = cfg.get("device", "cuda")
    a.seed = cfg.get("seed", 42)
    a.num_workers = cfg.get("num_workers", 10)
    a.pin_mem = True
    a.dataratio = "1.0"; a.stratified = False
    a.distributed = False; a.world_size = 1; a.gpu = 0; a.rank = 0; a.local_rank = -1
    a.output_dir = cfg["output_dir"]; a.log_dir = cfg.get("log_dir", cfg["output_dir"])
    a.task = cfg.get("task", "dr_retfound")
    a.norm = "IMAGENET"; a.enhance = False
    a.accum_iter = cfg["accum_iter"]
    return a


def set_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model_arch(args):
    """Build the ViT-L backbone via models_vit, per args.model (mirrors main_finetune.py).
    RETFound_mae -> random-init MAE ViT-L (weights loaded later via load_pretrained).
    RETFound_dinov2 -> timm DINOv2 ViT-L/14 (its base weights load here; RETFound teacher
    weights are applied later in load_pretrained)."""
    _ensure_repo_on_path()
    import models_vit as models
    if args.model == "RETFound_mae":
        model = models.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
    else:  # RETFound_dinov2 (patch14, img_size fixed at 224 inside the builder)
        model = models.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            args=args,
        )
    return model


def load_pretrained(model, args):
    """Download + load RETFound_mae_natureCFP (GATED) exactly as main_finetune does."""
    _ensure_repo_on_path()
    from huggingface_hub import hf_hub_download
    from util.pos_embed import interpolate_pos_embed
    from timm.models.layers import trunc_normal_

    ckpt_path = hf_hub_download(repo_id=f"YukunZhou/{args.finetune}",
                                filename=f"{args.finetune}.pth")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    # MAE checkpoint stores weights under "model"; DINOv2 under "teacher"
    checkpoint_model = checkpoint["teacher"] if args.model == "RETFound_dinov2" else checkpoint["model"]
    checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
    checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    if hasattr(model, "head") and hasattr(model.head, "weight"):
        trunc_normal_(model.head.weight, std=2e-5)
    return msg


def make_weighted_sampler(ds_train, nb_classes, minority_boost=None, seed=42):
    """WeightedRandomSampler for class-balanced batches (oversamples rare grades).

    Base per-sample weight = 1/class_count (so classes are seen ~equally). Optionally
    multiply a class's weight by minority_boost[c] to oversample it *further* than balance
    (e.g. {3: 2.0} to push R3 to 2x its balanced rate). NB: when using this sampler, use an
    UN-weighted loss (gamma-only focal / plain CE) to avoid double-correcting imbalance.
    """
    import numpy as np
    targets = np.array(ds_train.targets)
    counts = np.array([(targets == c).sum() for c in range(nb_classes)], dtype=float)
    counts = np.clip(counts, 1, None)
    class_w = 1.0 / counts
    if minority_boost:
        for c, b in minority_boost.items():
            class_w[c] *= b
    sample_w = class_w[targets]
    g = torch.Generator().manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(targets), replacement=True, generator=g)
    # expected per-class share of a drawn batch (for reporting)
    exp_share = (class_w * counts); exp_share = exp_share / exp_share.sum()
    return sampler, counts.astype(int), exp_share


def build_loaders(args, shuffle_train=True, train_sampler=None):
    """train/val/test loaders via RETFound's build_dataset (ImageFolder + transforms).

    If train_sampler is given (e.g. WeightedRandomSampler), it overrides shuffle."""
    _ensure_repo_on_path()
    from util.datasets import build_dataset
    ds_tr = build_dataset(is_train="train", args=args)
    ds_va = build_dataset(is_train="val", args=args)
    ds_te = build_dataset(is_train="test", args=args)
    dl_tr = torch.utils.data.DataLoader(
        ds_tr, batch_size=args.batch_size,
        shuffle=(shuffle_train and train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True)
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    dl_te = torch.utils.data.DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False)
    return (ds_tr, ds_va, ds_te), (dl_tr, dl_va, dl_te)


def class_weights_from_dataset(ds_train, nb_classes, device):
    """Inverse-frequency class weights (normalised to mean 1) for weighted CE."""
    import numpy as np
    targets = np.array(ds_train.targets)
    counts = np.array([(targets == c).sum() for c in range(nb_classes)], dtype=float)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (nb_classes * counts)   # inverse frequency
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device), counts.astype(int)


def build_optimizer(model, args):
    """Layer-wise LR-decay param groups (RETFound) + AdamW + native AMP scaler."""
    _ensure_repo_on_path()
    import util.lr_decay as lrd
    from util.misc import NativeScalerWithGradNormCount as NativeScaler

    eff_batch = args.batch_size * args.accum_iter
    if args.lr is None:
        args.lr = args.blr * eff_batch / 256
    no_wd = model.no_weight_decay() if hasattr(model, "no_weight_decay") else []
    groups = lrd.param_groups_lrd(model, weight_decay=args.weight_decay,
                                  no_weight_decay_list=no_wd, layer_decay=args.layer_decay)
    for g in groups:
        g["params"] = [p for p in g["params"] if p.requires_grad]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    scaler = NativeScaler()
    return optimizer, scaler
