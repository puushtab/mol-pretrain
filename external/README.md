# external/

Third-party code we run but do not own. Vendored as git submodules so the exact
commit is pinned. Nothing here is imported by `natural_compounds`.

| Repo | Why | Pinned commit |
|---|---|---|
| chemeleon | the pretraining recipe we reuse | `b5e5bd8`, 23 July 2026 |

After cloning the project:

```bash
git submodule update --init --recursive
```

`scripts/pretrain.py` calls `external/chemeleon/models/chemprop_mpnn/pretrain.py`
directly. Reading that file is worth ten minutes before touching anything here.
It takes the SMILES file with no header. It loads the standardisation vectors as
torch files from the working directory rather than from next to the Zarr store.

Never edit a submodule in place. If a patch is needed, fork it, commit there, and
repoint. Pretrained checkpoints go in `runs/`, never in git.
