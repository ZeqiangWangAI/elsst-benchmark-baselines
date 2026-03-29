import re
from pathlib import Path

from elsst_baselines.common.lora import discover_lora_target_modules, freeze_vision_modules

LEGACY_JSON_INSTRUCTION_RE = re.compile(
    r'Output a JSON array:\s*\[\{"term": "\.\.\.", "definition": "\.\.\."\}\]',
    flags=re.IGNORECASE,
)
PLAIN_TEXT_OUTPUT_INSTRUCTION = (
    "Output plain text only using `term: definition;` segments. "
    "Return between 1 and 5 concepts total, stop after the fifth concept, separate concepts with semicolons, "
    "keep definitions brief, and do not use JSON, bullets, numbering, or code fences."
)


def _torch_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_tokenizer_or_processor(model_name):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return tokenizer, tokenizer
    except Exception:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("processor does not expose a tokenizer for text-only usage")
        return tokenizer, processor


def _model_load_kwargs(qlora):
    kwargs = {
        "device_map": "auto",
        "torch_dtype": _torch_dtype(),
        "trust_remote_code": True,
    }
    if qlora:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_torch_dtype(),
        )
    return kwargs


def _read_target_modules(adapter_dir):
    target_modules_path = Path(adapter_dir) / "target_modules.txt"
    if not target_modules_path.exists():
        return None
    return [line.strip() for line in target_modules_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_adapter_name(adapter_name, fallback):
    import torch.nn as nn

    if adapter_name is None:
        return fallback
    normalized = re.sub(r"\W|^(?=\d)", "_", str(adapter_name).strip())
    if not normalized or normalized in dir(nn.Module):
        return fallback
    return normalized


def canonicalize_generation_prompt(prompt):
    prompt = str(prompt)
    if LEGACY_JSON_INSTRUCTION_RE.search(prompt):
        return LEGACY_JSON_INSTRUCTION_RE.sub(PLAIN_TEXT_OUTPUT_INSTRUCTION, prompt)
    return prompt


def load_generation_train_bundle(model_name, qlora=True, adapter_dir=None, adapter_name="default", ref_adapter_name=None):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModel, AutoModelForCausalLM

    tokenizer, processing_class = _load_tokenizer_or_processor(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = _model_load_kwargs(qlora)
    load_errors = []
    model = None
    candidate_loaders = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForVision2Seq

        candidate_loaders.append(AutoModelForVision2Seq)
    except ImportError:
        pass
    candidate_loaders.append(AutoModel)

    for loader in candidate_loaders:
        try:
            model = loader.from_pretrained(model_name, **kwargs)
            break
        except Exception as exc:
            load_errors.append(f"{loader.__name__}: {exc}")
    if model is None:
        raise RuntimeError("failed to load generation model: " + " | ".join(load_errors))

    freeze_vision_modules(model)
    if qlora:
        model = prepare_model_for_kbit_training(model)
    if adapter_dir:
        adapter_name = normalize_adapter_name(adapter_name, fallback="policy")
        ref_adapter_name = normalize_adapter_name(ref_adapter_name, fallback="reference") if ref_adapter_name else None
        if ref_adapter_name == adapter_name:
            ref_adapter_name = "reference"
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True, adapter_name=adapter_name)
        if ref_adapter_name:
            model.load_adapter(str(adapter_dir), adapter_name=ref_adapter_name)
        target_modules = _read_target_modules(adapter_dir) or discover_lora_target_modules(model)
    else:
        target_modules = discover_lora_target_modules(model)
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer, processing_class, target_modules


def load_generation_inference_bundle(model_name, adapter_dir=None, qlora=True):
    from peft import PeftModel
    from transformers import AutoModel, AutoModelForCausalLM

    tokenizer, processing_class = _load_tokenizer_or_processor(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs = _model_load_kwargs(qlora)
    load_errors = []
    model = None
    candidate_loaders = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForVision2Seq

        candidate_loaders.append(AutoModelForVision2Seq)
    except ImportError:
        pass
    candidate_loaders.append(AutoModel)

    for loader in candidate_loaders:
        try:
            model = loader.from_pretrained(model_name, **kwargs)
            break
        except Exception as exc:
            load_errors.append(f"{loader.__name__}: {exc}")
    if model is None:
        raise RuntimeError("failed to load generation model: " + " | ".join(load_errors))

    freeze_vision_modules(model)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    return model, tokenizer, processing_class


def save_generation_artifacts(model, tokenizer, output_dir, model_name, target_modules, merge_adapter=False, adapter_name=None):
    output_dir = Path(output_dir)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    if adapter_name and hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "base_model_name.txt").write_text(model_name + "\n", encoding="utf-8")
    (adapter_dir / "target_modules.txt").write_text("\n".join(target_modules) + "\n", encoding="utf-8")

    if merge_adapter and hasattr(model, "merge_and_unload"):
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)


def build_generation_prompt(prompt, tokenizer, disable_thinking=True):
    prompt = canonicalize_generation_prompt(prompt)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not disable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
    return prompt


def build_generation_training_text(prompt, response, tokenizer, disable_thinking=True):
    prompt = canonicalize_generation_prompt(prompt)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=not disable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
    return prompt + response
