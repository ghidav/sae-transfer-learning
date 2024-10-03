sudo apt install -y git-lfs
eval $(ssh-agent)
ssh-add ~/.ssh/id_hf
git lfs install
git clone git@hf.co:mech-interp/pythia-160m-deduped-rs-post
mkdir -p saes
mv pythia-160m-deduped-rs-post saes