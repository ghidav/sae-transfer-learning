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

# Computes logit difference between clean and corrupted answers tokens
def logits_diff(logits, correct_answer, incorrect_answer=None):
    correct_index = model.to_single_token(correct_answer)
    if incorrect_answer is None:
        return logits[0, -1, correct_index]
    else:
        incorrect_index = model.to_single_token(incorrect_answer)
        return logits[0, -1, correct_index] - logits[0, -1, incorrect_index]

# Hooks used for patching
def zero_abl_hook(x, hook, pos, head):
    x[:, pos, head] = 0
    return x

def patching_hook(x, hook, pos, head, corr):
    x[:, pos, head] = corr[:, pos, head]
    return x

def feature_patching_hook(x, hook, pos, head, f_in, f_out):
    x[:, pos, head] = x[:, pos, head] + f_in[None, pos, head] - f_out[None, pos, head]
    return x

# Base class for circuits
class TransformerCircuit:
    def __init__(self, model, task):
        self.model = model
        self.task = task

    def get_node(self, node_name):
        for node in self.task['nodes']:
            if node['name'] == node_name:
                return node
        return None

    def get_variable(self, variable_name):
        for variable in self.task['variables']:
            if variable['name'] == variable_name:
                return variable
        return None

    @classmethod
    def read_variable(self, x):
        if '+' in x:
            offset = int(x.split('+')[-1])
        elif '-' in x:
            offset = int(x.split('-')[-1])
        else:
            offset = 0

        pattern = r"\{([^}]*)\}"
        return re.findall(pattern, x)[0], offset

# Class for IOI circuit
class IOICircuit(TransformerCircuit):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.names = self.get_variable('IO')['values']
        self.components = ['q', 'k', 'v', 'z']

    def run_with_patch(
            self, 
            prompt: dict, 
            node_names: str, 
            attribute: str, 
            method: str ='zero',
            patches: list=None,
            verbose: bool=False):
        '''
        Function to run the circuit with a patch applied to the specified nodes.
        Takes in:
        - prompt: the prompt instance to be used
        - node_names: the names of the nodes to be patched
        - attribute: the attribute of the task to be patched
        - method: the method to be used for patching (one of 'zero', 'corr', 'feature')
        - patches: the patches to be applied to the nodes if using 'feature' method

        Returns the difference of answer logits between the clean and patched prompt.
        '''
        # Checks
        assert method in ['zero', 'corr', 'feature'], "Method must be either 'zero', 'corr', 'feature'"
        assert attribute in ['IO', 'S', 'Pos'], "Attribute must be either 'IO', 'S', 'Pos'"
        for node_name in node_names:
            node, components = node_name.split('.')
            assert self.get_node(node) is not None, f"Node {node_name} not found in the task"
            for c in components:
                assert c in self.components, f"Component {c} not found in the task"
        
        # Extracting variables from the prompt
        io = ' '+prompt['variables']['IO']
        s = ' '+prompt['variables']['S1']
        pos = prompt['variables']['Pos']
        pos = 0 if pos == "ABB" else 1

        clean_prompt = prompt['prompt']
        clean_tokens = model.to_tokens(clean_prompt)
        if verbose: print(f"Clean prompt: {clean_prompt}")

        new_io = None
        if method == 'corr':
            # Creation of the corrupted prompt based on the attribute
            if attribute == 'IO':
                # Pick a random new name to put in IO
                new_io = ' ' + random.choice(list(set(self.names) - {io[1:], s[1:]}))
                corr_prompt = clean_prompt.replace(io, new_io)
            elif attribute == 'S':
                # Pick a random new name to put in S
                new_s = ' ' + random.choice(list(set(self.names) - {io[1:], s[1:]}))
                corr_prompt = clean_prompt.replace(s, new_s)
            elif attribute == 'Pos':
                # Generate a new prompt with the opposite template
                new_template = self.task['templates'][1-pos]
                corr_prompt = new_template.format(IO=io[1:], S1=s[1:], S2=s[1:])

            corr_tokens = model.to_tokens(corr_prompt)
            if verbose: print(f"Corrupted prompt: {corr_prompt}")
            
            with torch.no_grad():
                _, corr_cache = self.model.run_with_cache(corr_tokens)

        with torch.no_grad():
            clean_logits = self.model(clean_tokens)
        
        # Hooking the model for patching
        hooks = []

        for name in node_names:
            # Split the node name into components (e.g. IH.qk -> IH, qk)
            node_name, components = name.split('.')
            node = self.get_node(node_name)

            # Add hooks to selected component of the node
            for component_name in components:
                if component_name in ['q', 'z']:
                    # Read the variable name and offset (e.g. IH.q -> S2+0 | PTH.z -> S1+1)
                    var, offset = self.read_variable(node['q'])
                else:
                    var, offset = self.read_variable(node['kv'])

                # Retrieve the token position of the variable
                var_pos = self.get_variable(var)['position'][pos] + offset

                # Add hooks to the component of each node head
                for head in node['heads']:
                    l, h = head.split('.')
                    hook_name = utils.get_act_name(component_name, int(l))
                    
                    if method == 'zero':
                        hook_fn = partial(zero_abl_hook, pos=var_pos, head=int(h))
                    elif method == 'corr':
                        hook_fn = partial(patching_hook, pos=var_pos, head=int(h), corr=corr_cache[hook_name])
                    elif method == 'feature':
                        f_in = patches[0][component_name][int(l)]
                        f_out = patches[1][component_name][int(l)]
                        hook_fn = partial(feature_patching_hook, pos=var_pos, head=int(h), f_in=f_in, f_out=f_out)

                    hooks.append((hook_name, hook_fn))
                    if verbose: print(f"Hooking L{l}H{h} {component_name} at position {var_pos}")

        with torch.no_grad():
            patched_logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)

        return new_io, clean_logits, patched_logits

device = "cuda"

# Load model and SAEs
model: HookedSAETransformer = HookedSAETransformer.from_pretrained("gpt2").to(device)
model.eval()

hook_z_sae = {}
hook_resid_pre_sae = {}
for layer in tqdm(range(12)):
    sae, cfg_dict, _ = SAE.from_pretrained(
        "gpt2-small-hook-z-kk",
        f"blocks.{layer}.hook_z",
        device=device
    )
    hook_z_sae[sae.cfg.hook_name] = sae

    sae, cfg_dict, _ = SAE.from_pretrained(
        "gpt2-small-res-jb",
        f"blocks.{layer}.hook_resid_pre",
        device=device
    )
    hook_resid_pre_sae[sae.cfg.hook_name] = sae

# Load task
with open('tasks/ioi/task.json') as f:
    task = json.load(f)

ioi_circuit = IOICircuit(model, task)
idx = 0

node_names = ['NMH.k'] #, 'NMH.q', 'bNMH.k']
attribute = 'IO'

example = task['prompts'][idx]

io = example['variables']['IO']
s = example['variables']['S2']

pos = example['variables']['Pos']
neg_pos = 'ABB' if pos == 'BAB' else 'BAB'

pos_id = 0 if pos == 'ABB' else 1

new_io, clean_logits, patched_logits = ioi_circuit.run_with_patch(
    example, 
    node_names=node_names,
    attribute=attribute,
    method='corr',
    verbose=True 
    )

print("\nClean logit diff: ", np.round(logits_diff(clean_logits, ' '+io, new_io).item(), 2))
print("Patched logit diff: ", np.round(logits_diff(patched_logits, ' '+io, new_io).item(), 2))

# SAE dictioanry
clean_prompt = example['prompt']
corr_prompt = clean_prompt.replace(' '+io, new_io)

clean_tokens = model.to_tokens(clean_prompt)
corr_tokens = model.to_tokens(corr_prompt)

with torch.no_grad():
    clean_logits = model(clean_tokens)

# Hooking the model for patching
def feature_z_patching_hook(x, hook, pos, f_in):
    x[:, pos] = x[:, pos] + f_in[None]
    return x

def feature_qkv_patching_hook(x, hook, pos, head, edit):
    x[:, pos, head] = edit
    return x

N = 100

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
            
            #with torch.no_grad():
            #    _, sae_cache = sae.run_with_cache(clean_acts)
            #sae_acts = sae_cache['hook_sae_acts_post'][None]

            W_dec = sae.W_dec.clone().detach()

            # Initialize influence vectors as leaf nodes
            f_in = torch.ones(W_dec.shape[0], device=W_dec.device, dtype=torch.float) / W_dec.shape[0]
            f_in.requires_grad = True

            transformed_acts = clean_acts + torch.matmul(f_in, W_dec)
            loss = torch.norm(transformed_acts - corr_acts)
            loss.backward()

            gradient_importance = f_in.grad.abs()
            best_features_ids = torch.argsort(gradient_importance, descending=True)[:N]
            best_features = W_dec[best_features_ids] # N x dm

            # Initialize alpha coefficients for the top 10 features
            alphas = torch.randn(N, device=W_dec.device, dtype=torch.float) / N
            alphas.requires_grad = True
            optimizer = torch.optim.SGD([alphas], lr=0.02)

            # Optimization loop
            for i in range(500):
                optimizer.zero_grad()

                transformed_acts = clean_acts + torch.matmul(alphas, best_features)
                
                loss = torch.norm(transformed_acts - corr_acts)
                loss.backward()
                optimizer.step()

            optimized_alphas = alphas.detach()

            if component_name == "z":
                z_vectors[l].append(torch.matmul(optimized_alphas, best_features))
            else:
                hook_name = utils.get_act_name(component_name, l)
                rs_pre = clean_cache[utils.get_act_name('resid_pre', l)][:, var_pos]
                if component_name == "q":
                    M = model.W_Q[l, h]
                elif component_name == "k":
                    M = model.W_K[l, h]
                elif component_name == "v":
                    M = model.W_V[l, h]
                edit = torch.matmul(rs_pre + torch.matmul(optimized_alphas, best_features)[None], M)
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