# Cluster setup (MIT ORCD Engaging)

## Once per account

```bash
# SSH key for GitHub, then add the public key at github.com -> Settings -> SSH keys
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github && cat ~/.ssh/id_ed25519_github.pub
ssh -T git@github.com

# uv, no sudo and no module needed
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

git clone --recurse-submodules git@github.com:puushtab/mol-pretrain.git ~/mol-pretrain
cd ~/mol-pretrain && source slurm/session.sh && make test
```

Do not run `conda init`. Do not `pip install --user`. Do not keep data in `$HOME`.

## Every session

```bash
cd ~/mol-pretrain && source slurm/session.sh
```

Login nodes have no GPU. `torch.cuda.is_available()` is `False` there by design.

## Storage

| Path | Size | Backed up | Use for |
|---|---|---|---|
| `$HOME` | 200 GB | yes | code, venv |
| `$HOME/orcd/scratch` | 1 TB | no, purged after six months idle | uv cache, data, runs |
| `$HOME/orcd/pool` | 1 TB | no | datasets to keep |

Check usage with `cat ~/orcd/.quota`. The Mordred store runs about 4 GB per
100k molecules. The size curve then needs roughly 8 GB. The full million would
need 40.

## Jobs

| Need | Command |
|---|---|
| Interactive GPU shell, two hours | `./slurm/gpu-shell.sh` |
| Descriptors, CPU array | `sbatch slurm/descriptors.sbatch data/processed/corpora/<name>.csv` |
| Pretraining, six hours | `sbatch slurm/pretrain.sbatch <corpus> <size> <seed>` |
| Queue | `squeue --me` |

| Partition | Wall | GPUs | Notes |
|---|---|---|---|
| `mit_normal` | 12 h | none | descriptors live here |
| `mit_normal_gpu` | 6 h | 2 | L40S by default, `-G h200:1` for 141 GB |
| `mit_preemptable` | 48 h | 4 | can be killed at any moment |

The size curve does not fit on `mit_normal_gpu`. The six-hour wall turns it into
about twenty queued jobs. It belongs on `mit_preemptable`, where the sizes run
one per GPU inside a single 48-hour allocation. Checkpoint and resume are then
mandatory, not optional.

Ask for cores rather than a second GPU. The reference run looks bound by graph
featurization on the CPU, not by the GPU. So `-c 16` with one GPU beats `-c 8`
with two.

## Open question

ORCD publishes no GPU-hour allocation for the base tier. One page warns that
over-requesting cores per GPU "may use up your allocation". So accounting exists,
and its size is undocumented. Asked at orcd-help-engaging@mit.edu, answer
pending. This is the only unknown that could break the plan rather than slow it.
