from sae_lens import HookedSAETransformer
from tqdm import tqdm
import json
import os
import torch
import pandas as pd

import time

os.environ["HF_HOME"] = "/workspace/huggingface"

PATHS = [
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/layer_0',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_1',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_2',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_3',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_4',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_5',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_6',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_7',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_8',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_9',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_10',
    '/workspace/huggingface/hub/models--ghidav--gpt2-sae-attn-tl/snapshots/54777f8ad8197017c54157a34152ac4da3e7da1c/transfer_layer_11',
]
    
from circuit import IOICircuit, IOIPrompt

device = "cuda"

def logits_diff(logits, correct_answer, incorrect_answer=None):
    correct_index = model.to_single_token(correct_answer)
    if incorrect_answer is None:
        return logits.cpu().numpy()[0, -1, correct_index]
    else:
        incorrect_index = model.to_single_token(incorrect_answer)
        return logits.cpu().numpy()[0, -1, correct_index] - logits.cpu().numpy()[0, -1, incorrect_index]

def logits_score(clean_logits, patched_logits, corr_logits, correct_answer, incorrect_answer=None):
    clean_score = logits_diff(clean_logits, correct_answer, incorrect_answer)
    patched_score = logits_diff(patched_logits, correct_answer, incorrect_answer)
    corr_score = logits_diff(corr_logits, correct_answer, incorrect_answer)

    score = (patched_score - clean_score) / (corr_score - clean_score)
    return score

# Load task
with open('tasks/ioi/cfg.json') as f:
    cfg = json.load(f)

with open('tasks/ioi/prompts.json') as f:
    prompts = json.load(f)

with open('tasks/ioi/names.json') as f:
    names = json.load(f)

print(f"N. prompts: {len(prompts)}")

# Load model and SAEs
model: HookedSAETransformer = HookedSAETransformer.from_pretrained("gpt2").to(device)
model.eval()

ioi_circuit = IOICircuit(
    model=model,
    cfg=cfg,
    prompts=prompts,
    names=names)

ioi_circuit.load_saes('resid_pre')
ioi_circuit.load_saes('z')
ioi_circuit.get_activations()

#for path in PATHS:
#    ioi_circuit.load_local_sae(path)

#all_nodes_labels = ['bNMH-q', 'bNMH-qk', 'IH+DTH-z', 'SIH-v', 'SIH-z', 'bNMH-z']
#all_nodes = [['bNMH.q'], ['bNMH.qk'], ['IH.z', 'DTH.z'], ['SIH.v'], ['SIH.z'], ['bNMH.z']]

all_nodes_labels = ['IH+DTH-z', 'SIH-z', 'bNMH-z']
all_nodes = [['IH.z', 'DTH.z'], ['SIH.z'], ['bNMH.z']]

for node_names, node_label in zip(all_nodes, all_nodes_labels):
    
    scores_df = {
        'clean_ld': [],
        'sae_full_ld': [],
        'sae_average_ld': [],
        'ablation_ld': []
    }

    for idx in tqdm(range(128)):
        try:
            example = IOIPrompt(prompts[idx], id=idx)
            example.tokenize(model)
            io = example.get_variable('IO')
            s = example.get_variable('S')

            # Get clean logits
            with torch.no_grad():
                clean_logits, clean_cache = ioi_circuit.model.run_with_cache(example.tokens)

            scores_df['clean_ld'].append(logits_diff(clean_logits, io, s).item())

            ### SAE FEATURES - FULL
            patched_logits = ioi_circuit.run_with_reconstruction(
                example, 
                node_names=node_names,
                method='sae',
                cache=clean_cache,
                reconstruction='sufficency',
                verbose=False
                )
        
            scores_df['sae_full_ld'].append(logits_diff(patched_logits, io, s).item())

            ### SAE FEATURES - AVERAGE
            patched_logits = ioi_circuit.run_with_reconstruction(
                example, 
                node_names=node_names,
                method='sae',
                cache=clean_cache,
                reconstruction='necessity',
                verbose=False
                )
        
            scores_df['sae_average_ld'].append(logits_diff(patched_logits, io, s).item())

            ### ABLATION
            patched_logits = ioi_circuit.run_with_reconstruction(
                example, 
                node_names=node_names,
                method='ablation',
                cache=clean_cache,
                verbose=False
                )
        
            scores_df['ablation_ld'].append(logits_diff(patched_logits, io, s).item())
        except Exception as e:
            print(f"Error in {idx}: {e}")

    scores_df = pd.DataFrame(scores_df)
    scores_df.to_json(f'tasks/ioi/sn-scores-standard/{node_label}.json')
