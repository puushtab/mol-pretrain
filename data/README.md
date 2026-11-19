# data/

Almost nothing here is committed. The exception is the lipophilicity benchmark
and its ten frozen splits, 512 KB, because Stage D has to pair against the test
molecules Stage B used and a regenerated split is not the same split. See
`data/processed/splits/lipophilicity/README.md`. Point `NC_DATA_DIR` at a
bigger disk if `./data` is not where the files should live.

```
data/
  raw/                      downloads, untouched
    pubchem/                cleaned_pubchem_1MM.smiles, CheMeleon's own corpus
    zinc/
    coconut/
    tasks/                  one CSV per benchmark
    candidates/             benchmarks the screen considered. lipophilicity.csv is in git
  processed/                reproducible from raw plus configs
    corpora/                <corpus>_n<size>_s<seed>.csv and its stats JSON
    descriptors/            <corpus>_n<size>_s<seed>.zarr and standardisation vectors
    splits/                 <task>/seed<k>.json. lipophilicity/ is in git
```

## Sources

| Source | What | Where |
|---|---|---|
| PubChem 1M | CheMeleon's pretraining corpus | Zenodo record 15733575, `cleaned_pubchem_1MM.smiles`, 43.9 MB |
| ZINC20 | general drug-like molecules | https://zinc20.docking.org |
| COCONUT 2.0 | natural products | https://coconut.naturalproducts.net |
| TDC | ADMET benchmarks | https://tdcommons.ai |
| MoleculeNet | classic benchmarks | https://moleculenet.org/datasets-1 |

The same Zenodo record holds `pubchem_1MM.zarr.tar.xz`, 3 GB, the descriptors for
the full corpus. Downloading it beats recomputing it for any size we do not
subsample ourselves.

Record the download date and version of every raw file in `docs/data_log.md`.
