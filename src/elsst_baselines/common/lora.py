from typing import Iterable


PREFERRED_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
)
EXCLUDED_PATTERNS = ("embed", "lm_head", "output", "score", "vision", "visual")


def _iter_module_names(model_or_module_names):
    if isinstance(model_or_module_names, Iterable) and not hasattr(model_or_module_names, "named_modules"):
        for item in model_or_module_names:
            if isinstance(item, str):
                yield item
        return

    for name, module in model_or_module_names.named_modules():
        if not name:
            continue
        class_name = module.__class__.__name__.lower()
        if "linear" not in class_name and not hasattr(module, "weight"):
            continue
        yield name


def _is_excluded(name):
    lowered = name.lower()
    return any(pattern in lowered for pattern in EXCLUDED_PATTERNS)


def discover_lora_target_modules(model_or_module_names):
    names = list(_iter_module_names(model_or_module_names))
    preferred = sorted({name.rsplit(".", 1)[-1] for name in names if name.endswith(PREFERRED_SUFFIXES) and not _is_excluded(name)})
    if preferred:
        return preferred

    fallback = sorted(
        {
            name.rsplit(".", 1)[-1]
            for name in names
            if not _is_excluded(name)
        }
    )
    if not fallback:
        raise ValueError("could not discover LoRA target modules")
    return fallback


def freeze_vision_modules(model):
    for name, parameter in model.named_parameters():
        if any(pattern in name.lower() for pattern in ("vision", "visual", "image", "mm_projector")):
            parameter.requires_grad = False
