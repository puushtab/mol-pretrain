# Cluster setup (MIT ORCD Engaging)

## Once per account

1. SSH key for GitHub, added at github.com → Settings → SSH keys:
   ```bash
   ssh-keygen -t ed25519 -C "orcd" -f ~/.ssh/id_ed25519_github && cat ~/.ssh/id_ed25519_github.pub
   printf 'Host github.com\n  IdentityFile ~/.ssh/id_ed25519_github\n  IdentitiesOnly yes\n' >> ~/.ssh/config && chmod 600 ~/.ssh/config
   ssh -T git@github.com
   ```
2. uv (no sudo, no module):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
   ```
3. Clone into home and set identity:
   ```bash
   git clone git@github.com:puushtab/natural-compounds.git ~/natural-compounds
   cd ~/natural-compounds && git config user.name "Your Name" && git config user.email "you@mit.edu"
   ```
4. Install and verify:
   ```bash
   source slurm/session.sh && make test
   ```

Do not run `conda init`, do not `pip install --user`, do not keep data in `$HOME`.

## Every session

```bash
cd ~/natural-compounds && source slurm/session.sh
```

This exports the scratch paths, pulls `main`, syncs the venv, and prints the state.
Login nodes have no GPU: `torch.cuda.is_available()` is `False` there by design.

## Storage

| Path | Size | Backed up | Use for |
|---|---|---|---|
| `$HOME` | 200 GB | yes | code, venv |
| `$HOME/orcd/scratch` | 1 TB | no, purged after 6 months idle | uv cache, data, runs |
| `$HOME/orcd/pool` | 1 TB | no | datasets to keep around |

Check usage: `cat ~/orcd/.quota`.

## GPUs

| Need | Command |
|---|---|
| Interactive shell, 1 L40S, 2 h | `./slurm/gpu-shell.sh` |
| Batch pretraining, 6 h max | `sbatch slurm/pretrain.sbatch configs/pretrain/default.yaml data/processed/corpora/<name>.csv` |
| Queue status | `squeue --me` |
| Cancel | `scancel <jobid>` |

Partitions: `mit_normal_gpu` (6 h, 2 GPUs max, L40S default, `-G h200:1` for 141 GB memory)
and `mit_preemptable` (48 h, 4 GPUs, can be killed). Never ask for more than 16 cores per GPU.
Any run longer than 6 h must checkpoint and resume.

Verify the GPU from inside an allocation:
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
