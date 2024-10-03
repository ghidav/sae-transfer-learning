import json
import random

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from sae_lens import SAE


class IdentitySAE(nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """

    def __init__(self, activation_dim=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x

    def decode(self, f):
        return f

    def forward(self, x, output_features=False, **kwargs):
        if output_features:
            return x, x
        else:
            return x


########################
### LOADING EXAMPLES ###
########################


def load_examples(dataset, num_examples, model, seed=12, pad_to_length=None, length=None):
    examples = []
    dataset_items = open(dataset).readlines()
    random.seed(seed)
    random.shuffle(dataset_items)
    for line in dataset_items:
        data = json.loads(line)
        clean_prefix = model.tokenizer(
            data["clean_prefix"],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).input_ids
        patch_prefix = model.tokenizer(
            data["patch_prefix"],
            return_tensors="pt",
            padding=False,
            add_special_tokens=False,
        ).input_ids
        clean_answer = model.tokenizer(
            data["clean_answer"],
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        ).input_ids
        patch_answer = model.tokenizer(
            data["patch_answer"],
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
        ).input_ids

        # only keep examples where answers are single tokens
        if clean_prefix.shape[1] != patch_prefix.shape[1]:
            continue
        # only keep examples where clean and patch inputs are the same length
        if clean_answer.shape[1] != 1 or patch_answer.shape[1] != 1:
            continue
        # if we specify a `length`, filter examples if they don't match
        if length and clean_prefix.shape[1] != length:
            continue
        # if we specify `pad_to_length`, left-pad all inputs to a max length
        prefix_length_wo_pad = clean_prefix.shape[1]
        if pad_to_length:
            model.tokenizer.padding_side = "right"
            pad_length = pad_to_length - prefix_length_wo_pad
            if pad_length < 0:  # example too long
                continue
            # left padding: reverse, right-pad, reverse
            clean_prefix = t.flip(
                F.pad(
                    t.flip(clean_prefix, (1,)),
                    (0, pad_length),
                    value=model.tokenizer.pad_token_id,
                ),
                (1,),
            )
            patch_prefix = t.flip(
                F.pad(
                    t.flip(patch_prefix, (1,)),
                    (0, pad_length),
                    value=model.tokenizer.pad_token_id,
                ),
                (1,),
            )

        example_dict = {
            "clean_prefix": clean_prefix,
            "patch_prefix": patch_prefix,
            "clean_answer": clean_answer.item(),
            "patch_answer": patch_answer.item(),
            "prefix_length_wo_pad": prefix_length_wo_pad,
        }
        examples.append(example_dict)
        if len(examples) >= num_examples:
            break

    return examples


####################
### LOADING SAES ###
####################


def get_sae_path(
    dict_path,
    model_name,
    ckpt,
    layer=None,
    direction: str = "baseline",
):
    if model_name in ["pythia-160m-deduped"]:
        return f"{dict_path}/{direction}/L{layer}/{ckpt}"
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def load_saes(
    dict_path,
    model_name,
    cfg,
    modules,
    ckpt="1B",
    device="cuda",
    debug=False,
    direction: str = "baseline",
    layer: int = 0,
):

    dictionaries = {}
    component = modules[0].split(".")[-1]
    print(f"Loading {component} dictionaries...")

    if "attn" in component:
        dim = cfg.d_head * cfg.n_heads
    else:
        dim = cfg.d_model

    if dict_path is None:
        dictionaries = {module: IdentitySAE(dim) for module in modules}
    else:
        paths = [get_sae_path(dict_path, model_name, ckpt, layer=layer, direction=direction)]
        for module, path in zip(modules, paths):
            if debug:
                print(module, "->", path)
            try:
                dictionaries[module] = SAE.load_from_pretrained(path, device=device, dtype="float32")
            except:
                print(f"Failed to load SAE at {path}. Using Id dict.")
                dictionaries[module] = IdentitySAE(dim)

    return dictionaries
