"""ADPE configuration and compatibility with existing experiment environments."""

import math
import os
import ast


def extend_cfg(cfg):
    from yacs.config import CfgNode as CN

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.TRAINER.NAME = "ADPE"
    cfg.TRAINER.ADPE = CN()
    cfg.TRAINER.ADPE.PREC = "fp32"

    cfg.MODEL.ADPE = CN()
    model = cfg.MODEL.ADPE
    model.DEEP = True
    model.DROPOUT = 0.0
    model.U1 = 10.0
    model.U2 = 2.5
    model.U3 = 1.5
    model.ALPHA = 0.4
    model.V2T_ALTERNATING_EPOCHS = 5
    model.T2V_ALTERNATING_EPOCHS = 1
    model.STARTING_PHASE = "train_vision"
    model.GPA_MEAN = 0.9
    model.GPA_STD = 0.2
    model.MAX_LEN_QUANTILE = 1.0
    model.POOLING = CN()
    model.POOLING.NORM = "normto1"
    model.ATTR_BRANCH_DROPOUT_P = 0.15
    model.ATTR_BRANCH_DROPOUT_RESCALE = False
    model.ATTR_DROPOUT_FUSION_CE_WEIGHT = 0.5
    model.DESCRIPTIONS_PATH = ""
    model.DESCRIPTION_SOURCE_DATASET = ""
    model.ATTR_ORDER = ""
    model.ATTR_COUNT = 0

    model.BRANCH_ABLATION = CN()
    model.BRANCH_ABLATION.ENABLED = False
    model.BRANCH_ABLATION.KEEP_RATIOS = "1.0,0.75,0.5,0.25,0.0"
    model.BRANCH_ABLATION.REPEATS = 50
    model.BRANCH_ABLATION.EXACT_MAX = 256
    model.BRANCH_ABLATION.MASK_CHUNK = 8
    model.BRANCH_ABLATION.SEED = 1
    model.BRANCH_ABLATION.RUN_NAME = "adpe"

    model.DPE = CN()
    model.DPE.ENABLED = True
    model.DPE.SHOT_CAPACITY = 3
    model.DPE.ALPHA = 6.0
    model.DPE.BETA = 5.0
    model.DPE.TEXT_LR = 0.0006
    model.DPE.IMAGE_LR = 0.0006
    model.DPE.ALIGN = 0.5
    model.DPE.ENTROPY_THRESHOLD = 0.1
    model.DPE.ATTR_AFFINITY_WEIGHT = 0.4

    cfg.OPTIM.VISION_MAX_EPOCH = 50
    cfg.OPTIM.VISION_LR = 0.004
    cfg.OPTIM.TEXT_MAX_EPOCH = 10
    cfg.OPTIM.TEXT_LR = 0.004


def _boolean(value):
    value = value.strip().lower()
    if value not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        raise ValueError("Expected a boolean environment value, got {!r}".format(value))
    return value in {"1", "true", "yes", "on"}


def apply_environment_overrides(cfg, environ=None):
    """Resolve environment aliases once, before CLI overrides and logging.

    ADPE_* is preferred. TAP_* and DPE_* are accepted for existing launchers;
    conflicting aliases are rejected instead of silently changing a run.
    """
    environ = os.environ if environ is None else environ
    model = cfg.MODEL.ADPE
    settings = [
        (model, "ATTR_BRANCH_DROPOUT_P", float),
        (model, "ATTR_BRANCH_DROPOUT_RESCALE", _boolean),
        (model, "ATTR_DROPOUT_FUSION_CE_WEIGHT", float),
        (model, "DESCRIPTIONS_PATH", str),
        (model, "DESCRIPTION_SOURCE_DATASET", str),
        (model, "ATTR_ORDER", str),
        (model, "ATTR_COUNT", int),
    ]
    settings.extend((model.DPE, "DPE_" + name, _boolean if name == "ENABLED"
                     else int if name == "SHOT_CAPACITY" else float)
                    for name in model.DPE)
    settings.extend((model.BRANCH_ABLATION, "BRANCH_ABLATION_" + name,
                     _boolean if name == "ENABLED" else str if name in {"KEEP_RATIOS", "RUN_NAME"} else int)
                    for name in model.BRANCH_ABLATION)
    for node, suffix, parse in settings:
        names = ["ADPE_" + suffix, "TAP_" + suffix]
        if suffix.startswith("DPE_"):
            names.append(suffix)
        values = [(name, parse(environ[name])) for name in names if name in environ]
        if not values:
            continue
        if any(value != values[0][1] for _, value in values[1:]):
            raise ValueError("Conflicting ADPE environment aliases: {}".format(values))
        field = suffix
        for prefix in ("DPE_", "BRANCH_ABLATION_"):
            if suffix.startswith(prefix):
                field = suffix[len(prefix):]
                break
        setattr(node, field, values[0][1])


def merge_cli_options(cfg, options):
    """Accept integer spellings such as rho=0 for registered float settings."""
    options = list(options or [])
    if len(options) % 2:
        raise ValueError("Configuration overrides must be KEY VALUE pairs")
    for index in range(0, len(options), 2):
        node = cfg
        try:
            for key in options[index].split("."):
                node = getattr(node, key)
            value = ast.literal_eval(options[index + 1])
        except (AttributeError, KeyError, ValueError, SyntaxError):
            continue
        if type(node) is float and type(value) is int:
            options[index + 1] = str(float(value))
    cfg.merge_from_list(options)


def validate_cfg(cfg):
    model = cfg.MODEL.ADPE
    if cfg.TRAINER.NAME != "ADPE":
        raise ValueError("This project registers --trainer ADPE; got {!r}".format(cfg.TRAINER.NAME))
    if cfg.TRAINER.ADPE.PREC not in {"fp32", "amp"}:
        raise ValueError("TRAINER.ADPE.PREC must be fp32 or amp")
    if cfg.MODEL.BACKBONE.NAME != "ViT-B/16":
        raise ValueError("The current ADPE prompt architecture requires ViT-B/16")
    if cfg.DATASET.SUBSAMPLE_CLASSES not in {"all", "base", "new"}:
        raise ValueError("DATASET.SUBSAMPLE_CLASSES must be all, base or new")
    for name in ("DROPOUT", "ALPHA", "ATTR_BRANCH_DROPOUT_P", "GPA_MEAN"):
        if not 0 <= getattr(model, name) <= 1:
            raise ValueError("MODEL.ADPE.{} must be in [0, 1]".format(name))
    if not 0 < model.MAX_LEN_QUANTILE <= 1:
        raise ValueError("MODEL.ADPE.MAX_LEN_QUANTILE must be in (0, 1]")
    if not math.isfinite(model.GPA_STD) or model.GPA_STD <= 0:
        raise ValueError("MODEL.ADPE.GPA_STD must be positive and finite")
    for name in ("U1", "U2", "ATTR_DROPOUT_FUSION_CE_WEIGHT"):
        if not math.isfinite(getattr(model, name)) or getattr(model, name) < 0:
            raise ValueError("MODEL.ADPE.{} must be finite and non-negative".format(name))
    if not math.isfinite(model.U3) or model.U3 <= 0:
        raise ValueError("MODEL.ADPE.U3 must be positive for text prompt training")
    if model.STARTING_PHASE not in {"train_vision", "train_text"}:
        raise ValueError("MODEL.ADPE.STARTING_PHASE must be train_vision or train_text")
    if model.ATTR_COUNT < 0:
        raise ValueError("MODEL.ADPE.ATTR_COUNT must be non-negative")
    for name in ("VISION_MAX_EPOCH", "TEXT_MAX_EPOCH"):
        if getattr(cfg.OPTIM, name) <= 0:
            raise ValueError("OPTIM.{} must be positive".format(name))
    for name in ("V2T_ALTERNATING_EPOCHS", "T2V_ALTERNATING_EPOCHS"):
        if getattr(model, name) <= 0:
            raise ValueError("MODEL.ADPE.{} must be positive".format(name))
    # GPA indexes separate arrays for the two phases, so their budgets must
    # match the actual alternating schedule, including an incomplete last cycle.
    remaining = cfg.OPTIM.VISION_MAX_EPOCH + cfg.OPTIM.TEXT_MAX_EPOCH
    phase = model.STARTING_PHASE
    counts = {"train_vision": 0, "train_text": 0}
    while remaining:
        length = model.V2T_ALTERNATING_EPOCHS if phase == "train_vision" else model.T2V_ALTERNATING_EPOCHS
        count = min(length, remaining)
        counts[phase] += count
        remaining -= count
        phase = "train_text" if phase == "train_vision" else "train_vision"
    if (counts["train_vision"], counts["train_text"]) != (
        cfg.OPTIM.VISION_MAX_EPOCH, cfg.OPTIM.TEXT_MAX_EPOCH
    ):
        raise ValueError("Vision/text epoch budgets do not match the alternating schedule: {}".format(counts))
    dpe = model.DPE
    if dpe.SHOT_CAPACITY <= 0:
        raise ValueError("MODEL.ADPE.DPE.SHOT_CAPACITY must be positive")
    for name in ("ALPHA", "BETA", "TEXT_LR", "IMAGE_LR", "ALIGN", "ENTROPY_THRESHOLD"):
        if not math.isfinite(getattr(dpe, name)) or getattr(dpe, name) < 0:
            raise ValueError("MODEL.ADPE.DPE.{} must be finite and non-negative".format(name))
    if not 0 <= dpe.ATTR_AFFINITY_WEIGHT <= 1:
        raise ValueError("MODEL.ADPE.DPE.ATTR_AFFINITY_WEIGHT must be in [0, 1]")
