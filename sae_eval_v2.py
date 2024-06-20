import torch
import numpy as np
from transformer_lens import utils
from sae_lens import HookedSAETransformer
from sae_lens import SAE
from tqdm import tqdm
import json
from functools import partial
import random
import re

from circuit import IOICircuit
from hooks import feature_z_patching_hook, feature_qkv_patching_hook
from tasks.ioi.dictionary import supervised_dictionary

device = "cuda"

def logits_diff(logits, correct_answer, incorrect_answer=None):
    correct_index = model.to_single_token(correct_answer)
    if incorrect_answer is None:
        return logits[0, -1, correct_index]
    else:
        incorrect_index = model.to_single_token(incorrect_answer)
        return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

# Load model and SAEs
model: HookedSAETransformer = HookedSAETransformer.from_pretrained("gpt2").to(device)
model.eval()

# Load task
with open('tasks/ioi/task.json') as f:
    task = json.load(f)

ioi_circuit = IOICircuit(model, task)
idx = 100

node_names = ['NMH.z', 'bNMH.z']
attribute = 'IO'

example = task['prompts'][idx]

io = example['variables']['IO']
s = example['variables']['S2']

pos = example['variables']['Pos']
neg_pos = 'ABB' if pos == 'BAB' else 'BAB'

pos_id = 0 if pos == 'ABB' else 1

# Corrupted
new_attr, clean_logits, patched_logits = ioi_circuit.run_with_patch(
    example, 
    node_names=node_names,
    attribute=attribute,
    method='corr',
    verbose=True 
    )

print("\nClean logit diff: ", np.round(logits_diff(clean_logits, ' '+io, ' '+new_attr).item(), 2))
print("Patched logit diff: ", np.round(logits_diff(patched_logits, ' '+io, ' '+new_attr).item(), 2))

# Supervised
bs = 64
prompts = [p['prompt'] for p in task['prompts']]
activations = {i: [] for i in ['q', 'k', 'v', 'z']}
for b in tqdm(range(0, len(prompts), bs)):
    tokens = model.to_tokens(prompts[b:b+bs])
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)

    for key in activations.keys():
        activations[key].append(cache.stack_activation(key))

activations = {key: torch.cat(values, 1) for key, values in activations.items()}
task_df = ioi_circuit.create_task_df()
io_vec, s_vec, pos_vec = supervised_dictionary(task_df, activations)

patches = []
for node in node_names:
    node, component = node.split('.')

    if attribute == 'IO':
        in_patch = io_vec[pos][new_attr]
        out_patch = s_vec[pos][io]
    elif attribute == 'S':
        in_patch = io_vec[pos][new_attr]
        out_patch = s_vec[pos][s]
    else:
        in_patch = pos_vec[pos]
        out_patch = pos_vec[neg_pos]

    patches.append((in_patch, out_patch))

_, clean_logits, patched_logits = ioi_circuit.run_with_patch(
    example, 
    node_names=node_names,
    attribute=attribute,
    method='feature',
    patches=patches,
    verbose=True
    )

print("\nClean logit diff: ", np.round(logits_diff(clean_logits, ' '+io, ' '+new_attr).item(), 2))
print("Patched logit diff: ", np.round(logits_diff(patched_logits, ' '+io, ' '+new_attr).item(), 2))

# SAE
N = 5

load_z = False
load_resid_pre = False
for node in node_names:
    if 'z' in node:
        load_z = True
    else:
        load_resid_pre = True

if load_z:
    ioi_circuit.load_saes('z')
if load_resid_pre:
    ioi_circuit.load_saes('resid_pre')
# TODO: Implement loading saes only for the reqired layers.

_, clean_logits, patched_logits = ioi_circuit.run_with_patch(
    example, 
    node_names=node_names,
    attribute=attribute,
    method='sae',
    verbose=True,
    new_attr=new_attr
    )

print("\nClean logit diff: ", np.round(logits_diff(clean_logits, ' '+io, ' '+new_attr).item(), 2))
print("Patched logit diff: ", np.round(logits_diff(patched_logits, ' '+io, ' '+new_attr).item(), 2))

"""
hooks = []

for name in node_names:
    # Split the node name into components (e.g. IH.qk -> IH, qk)
    node_name, components = name.split('.')
    node = ioi_circuit.get_node(node_name)
    print(f"\nHooking {components} of {node_name}...")

    # Add hooks to selected component of the node
    for component_name in components:
        
        z_vectors = [[] for l in range(model.cfg.n_layers)]
        
        if component_name in ['q', 'z']:
            # Read the variable name and offset (e.g. IH.q -> S2+0 | PTH.z -> S1+1)
            var, offset = ioi_circuit.read_variable(node['q'])
        else:
            var, offset = ioi_circuit.read_variable(node['kv'])

        # Retrieve the token position of the variable
        var_pos = ioi_circuit.get_variable(var)['position'][pos_id] + offset

        # Add hooks to the component of each node head
        for head in node['heads']:
            l, h = head.split('.')
            l, h = int(l), int(h)
            print(f"Hooking L{l}H{h} {component_name} at position {var_pos}")
            
            with torch.no_grad():
                _, clean_cache = model.run_with_cache(clean_tokens)
                _, corr_cache = model.run_with_cache(corr_tokens)

            if component_name == "z":
                hook_name = utils.get_act_name(component_name, l)
                sae = hook_z_sae[hook_name]
                clean_acts = torch.matmul(clean_cache[hook_name][:, var_pos, h], model.W_O[l, h]).detach()
                corr_acts = torch.matmul(corr_cache[hook_name][:, var_pos, h], model.W_O[l, h]).detach()
            else:
                hook_name = utils.get_act_name('resid_pre', l)
                sae = hook_resid_pre_sae[hook_name]
                clean_acts = clean_cache[hook_name][:, var_pos]
                corr_acts = corr_cache[hook_name][:, var_pos]
            
            # Feature search
            #feature_vector = optimization_fs(sae, clean_acts, corr_acts, var_pos, h, N)
            feature_vector = greedy_fs(sae, clean_acts, corr_acts, var_pos, h, N)

            if component_name == "z":
                z_vectors[l].append(feature_vector)
            else:
                hook_name = utils.get_act_name(component_name, l)
                rs_pre = clean_cache[utils.get_act_name('resid_pre', l)][:, var_pos]
                if component_name == "q":
                    M = model.W_Q[l, h]
                elif component_name == "k":
                    M = model.W_K[l, h]
                elif component_name == "v":
                    M = model.W_V[l, h]
                edit = torch.matmul(rs_pre + feature_vector[None], M)
                hook_fn = partial(feature_qkv_patching_hook, pos=var_pos, head=h, edit=edit)
                hooks.append((hook_name, hook_fn))
    
    # Add z hooks
    var, offset = ioi_circuit.read_variable(node['q'])
    var_pos = ioi_circuit.get_variable(var)['position'][pos_id] + offset
    
    for l in range(model.cfg.n_layers):
        if len(z_vectors[l]) > 0:
            z_vectors[l] = torch.stack(z_vectors[l], dim=0).sum(dim=0)
        
            hook_name = utils.get_act_name('attn_out', l)
            hook_fn = partial(feature_z_patching_hook, pos=var_pos, f_in=z_vectors[l])
            hooks.append((hook_name, hook_fn))

with torch.no_grad():
    patched_logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

print("\nClean logit diff: ", np.round(logits_diff(clean_logits, ' '+io, new_io).item(), 2))
print("Patched logit diff: ", np.round(logits_diff(patched_logits, ' '+io, new_io).item(), 2))
"""