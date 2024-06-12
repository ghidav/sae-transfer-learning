import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import numpy.random as random

tokenizer = AutoTokenizer.from_pretrained('gpt2')
names = ["John", "Mary", "Paul", "Anna", "Mark", "Lucy", "David", "Emma", "James", "Sara", "Lisa", "Brian", "Eric", "Jane", "Peter", "Susan", "Chris", "Nina", "Helen", "Diane", "Alice", "Bruce", "Kevin", "Linda", "Laura", "Megan", "Ryan", "Julie", "Steve", "Aaron", "Molly", "Cindy", "Grace", "Mason", "Ethan", "Chloe", "Claire", "Olivia", "Henry", "Nancy", "Maria", "Gary", "Karen", "Betty", "Shawn", "Holly", "Amber", "Tracy", "Judy"]

names_tokens = [tokenizer(' '+n, return_tensors='pt')['input_ids'][0] for n in names]
names = [n for n, t in zip(names, names_tokens) if len(t) == 1]

# IOI
def get_names():
    io, s1 = random.choice(np.arange(len(names)), size=2, replace=False)
    return names[io], names[s1]

abb_ioi_template = "When {IO} and {S1} went to the store, {S1} gave a drink to"
bab_ioi_template = "When {S1} and {IO} went to the store, {S1} gave a drink to"

prompts = []
for i in range(len(names)):
    for j in range(len(names)):
        if i != j:
            pos = random.rand()
            if pos > 0.5:
                ioi_template = abb_ioi_template
            else:
                ioi_template = bab_ioi_template
            
            io, s1 = names[i], names[j]
            prompt = ioi_template.format(IO=io, S1=s1)
            tokens = tokenizer(prompt, return_tensors='pt')['input_ids'][0]
            str_tokens = tokenizer.convert_ids_to_tokens(tokens)
                
            prompts.append({
                'prompt': prompt,
                'io': ' '+names[i],
                's1': ' '+names[j],
                'pos': 'ABB' if pos > 0.5 else 'BAB'
            })

pd.DataFrame(prompts).to_json('tasks/ioi.json', orient='records')
