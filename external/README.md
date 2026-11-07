# external/

Third-party code we read, patch, or run but do not own. Nothing in here is imported by
`natural_compounds`; it is vendored as **git submodules** so the exact commit is pinned.

```bash
# Add a repo (pin by commit, not branch)
git submodule add https://github.com/JacksonBurns/chemeleon external/chemeleon
git -C external/chemeleon checkout <commit>
git add external/chemeleon .gitmodules
git commit -m "Vendor chemeleon at <commit>"

# After cloning the project
git submodule update --init --recursive
```

| Repo | Why | Pinned commit |
|---|---|---|
| chemeleon | reference descriptor-pretraining recipe | TODO |

Rules:
- Never edit a submodule in place. If a patch is needed, fork, commit there, and repoint.
- Prefer a pip dependency when one exists (chemprop is installed via `uv sync --extra train`).
- Pretrained checkpoints go in `runs/` or the cluster, never in git.
