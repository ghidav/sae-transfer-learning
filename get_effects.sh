# Declare an associative array for lengths
declare -A lengths=( ["ioi"]=15 ["greater_than"]=12 ["subject_verb"]=6 )

# BASELINE
for task in "${!lengths[@]}"; do
    # Get the length for the current task
    len=${lengths[$task]}
    
    # Loop over layers 0 to 11
    for layer in {0..11}; do
        python effects.py -d "$task" -c resid_post -n 1024 -l "$len" -m pythia-160m-deduped -mt attrib --dict_path saes --device cuda --layer "$layer"
    done
done

# FORWARD
for ckpt in "100M" "200M" "300M" "400M" "500M"; do
    # Loop over the tasks
    for task in "${!lengths[@]}"; do
        # Get the length for the current task
        len=${lengths[$task]}
        
        # Loop over layers 0 to 11
        for layer in {1..11}; do
            python effects.py -d "$task" -c resid_post -n 1024 -l "$len" -m pythia-160m-deduped -mt attrib --dict_path saes --device cuda --layer "$layer" --direction forward --ckpt "$ckpt"
        done
    done
done

# BACKWARD
for ckpt in "100M" "200M" "300M" "400M" "500M"; do
    # Loop over the tasks
    for task in "${!lengths[@]}"; do
        # Get the length for the current task
        len=${lengths[$task]}
        
        # Loop over layers 0 to 11
        for layer in {0..10}; do
            python effects.py -d "$task" -c resid_post -n 1024 -l "$len" -m pythia-160m-deduped -mt attrib --dict_path saes --device cuda --layer "$layer" --direction backward --ckpt "$ckpt"
        done
    done
done