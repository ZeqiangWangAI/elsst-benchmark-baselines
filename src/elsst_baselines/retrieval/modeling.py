from pathlib import Path

from elsst_baselines.common.lora import discover_lora_target_modules


def _torch_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_sentence_transformer(model_name, max_seq_length):
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"device_map": "auto"}
    tokenizer_kwargs = {"padding_side": "left"}
    try:
        import flash_attn  # noqa: F401

        model_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    model = SentenceTransformer(
        model_name,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    model.max_seq_length = max_seq_length
    return model


def _sentence_transformer_backbone(model):
    first_module = model._first_module()
    if hasattr(first_module, "auto_model"):
        return first_module, first_module.auto_model
    raise RuntimeError("could not locate the transformer backbone inside SentenceTransformer")


def load_retrieval_train_bundle(model_name, max_seq_length):
    from peft import LoraConfig, TaskType, get_peft_model

    model = _load_sentence_transformer(model_name, max_seq_length=max_seq_length)
    first_module, backbone = _sentence_transformer_backbone(model)
    target_modules = discover_lora_target_modules(backbone)

    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()

    peft_task_type = getattr(TaskType, "FEATURE_EXTRACTION", None)
    if peft_task_type is None:
        raise RuntimeError("installed PEFT does not expose TaskType.FEATURE_EXTRACTION")

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=peft_task_type,
        target_modules=target_modules,
    )
    first_module.auto_model = get_peft_model(backbone, config)
    return model, target_modules


def load_retrieval_inference_model(model_name, max_seq_length, adapter_dir=None):
    from peft import PeftModel

    model = _load_sentence_transformer(model_name, max_seq_length=max_seq_length)
    if adapter_dir:
        first_module, backbone = _sentence_transformer_backbone(model)
        first_module.auto_model = PeftModel.from_pretrained(backbone, str(adapter_dir))
    return model


def save_retrieval_artifacts(model, output_dir, model_name, target_modules, merge_adapter=False):
    output_dir = Path(output_dir)
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    first_module, backbone = _sentence_transformer_backbone(model)
    backbone.save_pretrained(adapter_dir)
    first_module.tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "base_model_name.txt").write_text(model_name + "\n", encoding="utf-8")
    (adapter_dir / "target_modules.txt").write_text("\n".join(target_modules) + "\n", encoding="utf-8")

    if merge_adapter and hasattr(backbone, "merge_and_unload"):
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = backbone.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        first_module.tokenizer.save_pretrained(merged_dir)
