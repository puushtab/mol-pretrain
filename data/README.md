# data/

Nothing here is committed except this file and `.gitkeep`. Large files live locally
(or on the cluster; point `NC_DATA_DIR` in `.env` at them).

```
data/
  raw/          # downloads, untouched. One subfolder per source.
    zinc/
    coconut/
    tasks/      # TDC / MoleculeNet benchmark CSVs, one per task, with a `smiles` column
  processed/    # produced by scripts/, reproducible from raw + configs
    corpora/    # <name>.csv + <name>.stats.json
    splits/     # <task>/seed<k>.json (frozen scaffold splits)
```

## Sources

| Source | What | Where |
|---|---|---|
| ZINC20 | general drug-like molecules | https://zinc20.docking.org |
| COCONUT 2.0 | natural products | https://coconut.naturalproducts.net |
| TDC | ADMET benchmarks | `pip install PyTDC` or https://tdcommons.ai |
| MoleculeNet | classic benchmarks | https://moleculenet.org/datasets-1 |

Record the download date and version of every raw file in `docs/data_log.md`.
