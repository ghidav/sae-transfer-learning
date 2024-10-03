# Attribution Patching
for what in "faithfulness" "completeness"; do
    for task in "ioi" "greater_than" "subject_verb"; do
        for layer in {0..11}; do
            python faithfulness.py -c resid_post -n 1024 -m attrib -w $what --layer $layer --task $task --direction baseline
        done
    done

    for ckpt in "100M" "200M" "300M" "400M" "500M"; do
        for task in "ioi" "greater_than" "subject_verb"; do
            for layer in {1..11}; do
                python faithfulness.py -c resid_post -n 1024 -m attrib -w $what --layer $layer --task $task --direction forward --ckpt $ckpt
            done
        done
    done

    for ckpt in "100M" "200M" "300M" "400M" "500M"; do
        for task in "ioi" "greater_than" "subject_verb"; do
            for layer in {0..10}; do
                python faithfulness.py -c resid_post -n 1024 -m attrib -w $what --layer $layer --task $task --direction backward --ckpt $ckpt
            done
        done
    done
done