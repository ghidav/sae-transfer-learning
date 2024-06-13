import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import numpy.random as random
import argparse
import json

# Argument parser
parser = argparse.ArgumentParser(description='Generate tasks for SAE transfer learning')
parser.add_argument('-m', '--model', type=str, default='ioi', help='Model to generate tasks for.')
parser.add_argument('-t', '--task', type=str, default='ioi', help='Task to generate.')
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model)

with open(f'tasks/{args.task.lower()}/cfg.json', 'r') as f:
    base_cfg = json.load(f)

# IOI
if args.tasks == 'ioi':
    names = base_cfg['variables'][0]['values']
    names_tokens = [tokenizer(' '+n, return_tensors='pt')['input_ids'][0] for n in names]
    names = [n for n, t in zip(names, names_tokens) if len(t) == 1]
    
    templates = base_cfg['templates']

    prompts = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                pos = random.rand()
                if pos > 0.5:
                    ioi_template = templates[0] # ABB
                else:
                    ioi_template = templates[1] # BAB
                
                io, s1 = names[i], names[j]
                prompt = ioi_template.format(IO=io, S1=s1)
                tokens = tokenizer(prompt, return_tensors='pt')['input_ids'][0]
                str_tokens = tokenizer.convert_ids_to_tokens(tokens)
                    
                prompts.append({
                    "prompt": prompt,
                    "variables": [
                        {
                            "name": "IO",
                            "value": names[i],
                            "pos": 2 if pos > 0.5 else 4
                        },
                        {
                            "name": "S1",
                            "value": names[j],
                            "pos": 4 if pos > 0.5 else 2
                        },
                        {
                            "name": "S2",
                            "value": names[j],
                            "pos": 10
                        }
                ]})

    base_cfg['prompts'] = prompts
    # Write updated config
    with open(f'tasks/{args.task.lower()}/task.json', 'w') as f:
        json.dump(base_cfg, f, indent=4)
