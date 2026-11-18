# Stage C result tables

Copies, taken on 15 November 2026, of the small tables the scripts write under
`runs/`. The cells themselves (one `predictions.csv` and `manifest.json` each)
stay on the cluster; these tables are what the figures in `docs/figures/` are
drawn from, and each can be rebuilt from the cells with the command beside it.

| File | What it holds | Rebuilt by |
|---|---|---|
| `results_per_run.csv` | one row per fine-tuning cell of record: 500 and all labels from `runs/finetune/`, 50 and 100 from the Amendment 16 rerun in `runs/finetune_a16/` | `scripts/campaign_of_record.py`, then `scripts/evaluate.py runs/finetune_record --baseline scratch --baseline chemprop_descriptors --baseline chemprop` |
| `results_summary.csv`, `paired_differences__vs_*.csv` | means and paired differences from the same cells | the same |
| `finetune_size_curve.csv` | the numbers behind `finetune_size_curve*.png`: mean, paired difference from scratch and its 95% interval, per task, label count and arm | `scripts/plot_size_curve.py finetune` |
| `finetune_size_curve_registered.csv` | the same with the 50 and 100 label cells as first registered (before Amendment 16) | `scripts/plot_size_curve.py finetune --low-labels registered` |
| `exposure_sensitivity.csv` | Stage C task 14: the encoder pretrained with no benchmark molecule in its corpus against the reference, beside the seed-to-seed difference | `scripts/exposure_sensitivity.py` |
| `probe_full_labels_paired.csv`, `probe_full_labels_paired_clipped.csv` | linear probe at full labels, paired against the random encoder, without and with the cap on standardised features | `scripts/plot_size_curve.py probe`, and the same with `--runs runs/probe_clipped --suffix _clipped` |

Intervals are 95% t-intervals on five paired per-split differences, the two
pretraining seeds averaged within a split first. They are not adjusted for the
number of comparisons.
