from typing import Any, cast

import torch
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule
from transformers import GPTNeoXForCausalLM, AutoTokenizer, Trainer, TrainingArguments, PushToHubCallback, DataCollatorForLanguageModeling

import sys
sys.path.append('sae-transfer-learning')
from convert_model import convert_model

from huggingface_hub import HfFolder
HfFolder.save_token('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')

def load_model(
    model_class_name: str,
    model_name: str,
    device: str | torch.device | None = None,
    model_from_pretrained_kwargs: dict[str, Any] | None = None,
) -> HookedRootModule:
    model_from_pretrained_kwargs = model_from_pretrained_kwargs or {}
    if model_name == 'marco-molinari_pythia-70m-instruct':
        fine_tuned_model = GPTNeoXForCausalLM.from_pretrained("marco-molinari/pythia-70m-instruct")
        base_model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped")
        pythia_cfg = base_model.cfg
        return convert_model(fine_tuned_model, pythia_cfg).to(device)
    if model_name == 'EleutherAI_pythia-70m-deduped':
        return HookedTransformer.from_pretrained(
            model_name='EleutherAI/pythia-70m-deduped', device=device, **model_from_pretrained_kwargs
        )
    if model_class_name == "HookedTransformer":
        return HookedTransformer.from_pretrained(
            model_name=model_name, device=device, **model_from_pretrained_kwargs
        )
    elif model_class_name == "HookedMamba":
        try:
            from mamba_lens import HookedMamba
        except ImportError:
            raise ValueError(
                "mamba-lens must be installed to work with mamba models. This can be added with `pip install sae-lens[mamba]`"
            )
        # HookedMamba has incorrect typing information, so we need to cast the type here
        return cast(
            HookedRootModule,
            HookedMamba.from_pretrained(
                model_name, device=cast(Any, device), **model_from_pretrained_kwargs
            ),
        )
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")
