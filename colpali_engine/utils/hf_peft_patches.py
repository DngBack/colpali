"""
Compatibility fixes for Hugging Face Transformers 5.5+ when loading ColVision PEFT adapters.

1) PEFT MoE conversion wrongly runs for paligemma (maps to \"llava\") → KeyError.
2) load_adapter() calls get_model_conversion_mapping(self) without key_mapping, so class-level
   _checkpoint_conversion_mapping (e.g. custom_text_proj LoRA rename for ColPali) is skipped,
   leaving adapter weights unloaded / on meta device.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_hf_peft_patches() -> None:
    """Idempotent; safe to call multiple times."""
    global _PATCHED
    if _PATCHED:
        return

    _patch_peft_moe_skip_non_moe()
    _patch_peft_adapter_uses_class_key_mapping()

    _PATCHED = True
    logger.debug("colpali_engine: Hugging Face PEFT compatibility patches applied")


def _patch_peft_moe_skip_non_moe() -> None:
    try:
        import transformers.integrations.peft as peft_mod
        from transformers.conversion_mapping import _MODEL_TO_CONVERSION_PATTERN
    except Exception as e:
        logger.warning("Could not apply PEFT MoE paligemma patch: %s", e)
        return

    if getattr(peft_mod, "_colpali_moe_peft_patch_applied", False):
        return
    _orig = peft_mod._convert_peft_config_moe

    def _safe_convert_peft_config_moe(peft_config, model_type: str):
        mapped = _MODEL_TO_CONVERSION_PATTERN.get(model_type)
        if mapped is not None and mapped not in peft_mod._MOE_TARGET_MODULE_MAPPING:
            return peft_config
        return _orig(peft_config, model_type)

    peft_mod._convert_peft_config_moe = _safe_convert_peft_config_moe
    peft_mod._colpali_moe_peft_patch_applied = True


def _patch_peft_adapter_uses_class_key_mapping() -> None:
    try:
        import transformers.integrations.peft as peft_mod
    except Exception as e:
        logger.warning("Could not apply PEFT adapter key_mapping patch: %s", e)
        return

    if getattr(peft_mod, "_colpali_adapter_key_mapping_patch_applied", False):
        return
    _orig = peft_mod.get_model_conversion_mapping

    def _get_model_conversion_mapping_with_class_key_mapping(
        model,
        key_mapping=None,
        hf_quantizer=None,
        add_legacy=True,
    ):
        if key_mapping is None:
            cls = model.__class__
            km = getattr(cls, "_checkpoint_conversion_mapping", None)
            if km:
                key_mapping = km
        return _orig(model, key_mapping, hf_quantizer, add_legacy)

    peft_mod.get_model_conversion_mapping = _get_model_conversion_mapping_with_class_key_mapping
    peft_mod._colpali_adapter_key_mapping_patch_applied = True
