# Corpus exposure to every benchmark, on the connectivity key

The worst exposure in the campaign is **coconut_noleak_n25000_s0** against **np_origin**: 4864 of its 25000 molecules are benchmark molecules, 19.456 percent of the corpus.

Read the other way round, the most thoroughly seen benchmark is **vapor_pressure** against **released_checkpoint_pool**: 97.9307 percent of its 2368 molecules are in that pool.

Measured with `natural_compounds.chem.connectivity_key`, the canonical SMILES with stereochemistry removed from both sides, counted per corpus row. Membership of the published corpus file is an upper bound on the encoder having taken a gradient step on the molecule. No label leaks; pretraining is unsupervised descriptor regression. The key merges genuine stereoisomers, so these are upper bounds on exact-structure identity. Regenerate with `uv run python scripts/exclusion_audit.py`.

**The PubChem arms were never excluded against np_origin.** The benchmark was cut from COCONUT after those draws were taken, so the COCONUT arms carry a 19.26 percent hole punched for it and the PubChem arms carry none. Their np_origin figures below are what was left in, not what survived a filter. No arm of any corpus was ever excluded against vapor_pressure or ptp1b, which did not exist when any draw was taken.

155 of 200 (corpus, benchmark) pairs have any overlap at all. The full table is in `exclusion-audit.csv`.

## Which contrasts are asymmetric

- **vapor_pressure**: pubchem_nearbench_noleak is the most exposed arm at 1.117 percent of its molecules, coconut_noleak the least exposed non-zero arm at 0.184 percent, a factor of 6. Clean at zero: zinc_general, zinc_general_noleak, zinc_nplike, zinc_nplike_noleak.
- **ptp1b**: coconut_noleak is the most exposed arm at 0.203 percent of its molecules, pubchem the least exposed non-zero arm at 0.017 percent, a factor of 12. Clean at zero: zinc_general, zinc_general_noleak, zinc_nplike, zinc_nplike_noleak.
- **np_origin**: coconut is the most exposed arm at 19.225 percent of its molecules, zinc_general the least exposed non-zero arm at 0.005 percent, a factor of 3845. Clean at zero: coconut_noleakskel.
- **bbbp**: pubchem_nearbench_noleak is the most exposed arm at 1.207 percent of its molecules, pubchem_noleak the least exposed non-zero arm at 0.112 percent, a factor of 11. Clean at zero: zinc_general, zinc_general_noleak, zinc_nplike, zinc_nplike_noleak.
- **esol**: pubchem_nearbench_noleak is the most exposed arm at 0.472 percent of its molecules, pubchem_noleak the least exposed non-zero arm at 0.043 percent, a factor of 11. Clean at zero: zinc_general, zinc_general_noleak, zinc_nplike, zinc_nplike_noleak.

## Every pair, worst share of corpus first

| corpus | benchmark | corpus rows | benchmark rows | overlap | % of corpus | % of benchmark |
|---|---|---|---:|---:|---:|---:|
| coconut_noleak_n25000_s0 | np_origin | 25000 | 60170 | 4864 | 19.456 | 7.5619 |
| coconut_n20000_s0 | np_origin | 20000 | 60170 | 3845 | 19.225 | 6.0545 |
| coconut_noleak_n25000_s2 | np_origin | 25000 | 60170 | 4770 | 19.08 | 7.3907 |
| coconut_noleak_n25000_s1 | np_origin | 25000 | 60170 | 4719 | 18.876 | 7.3093 |
| pubchem_nearbench_noleak_n25000_s0 | np_origin | 25000 | 60170 | 641 | 2.564 | 0.8459 |
| pubchem_nearbench_noleak_n25000_s2 | np_origin | 25000 | 60170 | 640 | 2.56 | 0.8842 |
| pubchem_nearbench_noleak_n25000_s1 | np_origin | 25000 | 60170 | 624 | 2.496 | 0.841 |
| pubchem_n500_s0 | np_origin | 500 | 60170 | 12 | 2.4 | 0.0199 |
| pubchem_n1000_s0 | np_origin | 1000 | 60170 | 20 | 2.0 | 0.0332 |
| pubchem_n1000_s1 | np_origin | 1000 | 60170 | 18 | 1.8 | 0.0283 |
| pubchem_n500_s1 | np_origin | 500 | 60170 | 9 | 1.8 | 0.015 |
| pubchem_n5000_s1 | np_origin | 5000 | 60170 | 89 | 1.78 | 0.1463 |
| pubchem_n10000_s1 | np_origin | 10000 | 60170 | 169 | 1.69 | 0.2792 |
| pubchem_noleak_n25000_s2 | np_origin | 25000 | 60170 | 392 | 1.568 | 0.6448 |
| pubchem_n50000_s0 | np_origin | 50000 | 60170 | 782 | 1.564 | 1.2747 |
| pubchem_n25000_s1 | np_origin | 25000 | 60170 | 390 | 1.56 | 0.6399 |
| pubchem_n100000_s0 | np_origin | 100000 | 60170 | 1559 | 1.559 | 2.4713 |
| released_checkpoint_pool | np_origin | 944286 | 60170 | 14640 | 1.5504 | 17.1697 |
| pubchem_n20000_s0 | np_origin | 20000 | 60170 | 308 | 1.54 | 0.5052 |
| pubchem_n100000_s1 | np_origin | 100000 | 60170 | 1536 | 1.536 | 2.4265 |
| pubchem_n50000_s1 | np_origin | 50000 | 60170 | 767 | 1.534 | 1.2531 |
| pubchem_n25000_s0 | np_origin | 25000 | 60170 | 382 | 1.528 | 0.6266 |
| pubchem_noleak_n100000_s0 | np_origin | 100000 | 60170 | 1526 | 1.526 | 2.3783 |
| pubchem_noleak_n25000_s0 | np_origin | 25000 | 60170 | 380 | 1.52 | 0.6166 |
| pubchem_n5000_s0 | np_origin | 5000 | 60170 | 75 | 1.5 | 0.1246 |
| pubchem_n10000_s0 | np_origin | 10000 | 60170 | 146 | 1.46 | 0.2426 |
| pubchem_noleak_n25000_s1 | np_origin | 25000 | 60170 | 345 | 1.38 | 0.5684 |
| pubchem_nearbench_noleak_n25000_s0 | bbbp | 25000 | 2050 | 316 | 1.264 | 12.2376 |
| pubchem_nearbench_noleak_n25000_s1 | bbbp | 25000 | 2050 | 316 | 1.264 | 12.2888 |
| pubchem_nearbench_noleak_n25000_s2 | vapor_pressure | 25000 | 2368 | 302 | 1.208 | 11.951 |
| pubchem_nearbench_noleak_n25000_s1 | vapor_pressure | 25000 | 2368 | 273 | 1.092 | 10.473 |
| pubchem_nearbench_noleak_n25000_s2 | bbbp | 25000 | 2050 | 273 | 1.092 | 10.9063 |
| pubchem_nearbench_noleak_n25000_s0 | vapor_pressure | 25000 | 2368 | 263 | 1.052 | 10.3463 |
| pubchem_nearbench_noleak_n25000_s2 | esol | 25000 | 1128 | 124 | 0.496 | 6.9955 |
| pubchem_nearbench_noleak_n25000_s1 | esol | 25000 | 1128 | 121 | 0.484 | 6.7265 |
| pubchem_nearbench_noleak_n25000_s0 | esol | 25000 | 1128 | 109 | 0.436 | 6.0987 |
| pubchem_n5000_s1 | vapor_pressure | 5000 | 2368 | 20 | 0.4 | 0.8446 |
| pubchem_n1000_s0 | vapor_pressure | 1000 | 2368 | 4 | 0.4 | 0.1689 |
| pubchem_n500_s0 | vapor_pressure | 500 | 2368 | 2 | 0.4 | 0.0845 |
| pubchem_n10000_s1 | vapor_pressure | 10000 | 2368 | 38 | 0.38 | 1.5625 |
| coconut_noleakskel_n5000_s1 | bbbp | 5000 | 2050 | 19 | 0.38 | 0.9729 |
| coconut_n20000_s0 | bbbp | 20000 | 2050 | 69 | 0.345 | 3.0722 |
| coconut_noleakskel_n25000_s1 | bbbp | 25000 | 2050 | 86 | 0.344 | 4.1475 |
| coconut_noleakskel_n25000_s0 | bbbp | 25000 | 2050 | 83 | 0.332 | 4.0451 |
| coconut_noleakskel_n5000_s1 | vapor_pressure | 5000 | 2368 | 16 | 0.32 | 0.6757 |
| pubchem_n25000_s0 | bbbp | 25000 | 2050 | 79 | 0.316 | 3.9427 |
| pubchem_n20000_s0 | bbbp | 20000 | 2050 | 63 | 0.315 | 3.1746 |
| coconut_noleakskel_n5000_s0 | bbbp | 5000 | 2050 | 15 | 0.3 | 0.768 |
| pubchem_n50000_s1 | bbbp | 50000 | 2050 | 146 | 0.292 | 7.3221 |
| pubchem_n50000_s0 | bbbp | 50000 | 2050 | 145 | 0.29 | 7.1685 |
| pubchem_n100000_s0 | bbbp | 100000 | 2050 | 285 | 0.285 | 13.4665 |
| pubchem_n100000_s1 | bbbp | 100000 | 2050 | 284 | 0.284 | 13.5689 |
| pubchem_n25000_s1 | bbbp | 25000 | 2050 | 71 | 0.284 | 3.5842 |
| pubchem_n25000_s1 | vapor_pressure | 25000 | 2368 | 70 | 0.28 | 2.9139 |
| pubchem_n10000_s0 | bbbp | 10000 | 2050 | 28 | 0.28 | 1.4337 |
| pubchem_n10000_s1 | bbbp | 10000 | 2050 | 28 | 0.28 | 1.4337 |
| pubchem_n5000_s0 | vapor_pressure | 5000 | 2368 | 14 | 0.28 | 0.5912 |
| pubchem_n100000_s1 | vapor_pressure | 100000 | 2368 | 279 | 0.279 | 11.402 |
| pubchem_n50000_s1 | vapor_pressure | 50000 | 2368 | 139 | 0.278 | 5.6588 |
| released_checkpoint_pool | bbbp | 944286 | 2050 | 2607 | 0.2761 | 81.874 |
| released_checkpoint_pool | vapor_pressure | 944286 | 2368 | 2605 | 0.2759 | 97.9307 |
| pubchem_n100000_s0 | vapor_pressure | 100000 | 2368 | 270 | 0.27 | 11.1909 |
| pubchem_n50000_s0 | vapor_pressure | 50000 | 2368 | 135 | 0.27 | 5.6588 |
| coconut_noleakskel_n5000_s0 | vapor_pressure | 5000 | 2368 | 13 | 0.26 | 0.549 |
| coconut_noleakskel_n25000_s2 | bbbp | 25000 | 2050 | 64 | 0.256 | 3.2258 |
| pubchem_n20000_s0 | vapor_pressure | 20000 | 2368 | 50 | 0.25 | 2.0693 |
| pubchem_n25000_s0 | vapor_pressure | 25000 | 2368 | 61 | 0.244 | 2.5338 |
| coconut_noleak_n25000_s1 | ptp1b | 25000 | 460 | 60 | 0.24 | 11.0092 |
| coconut_noleakskel_n25000_s1 | vapor_pressure | 25000 | 2368 | 60 | 0.24 | 2.4493 |
| pubchem_n10000_s0 | vapor_pressure | 10000 | 2368 | 24 | 0.24 | 1.0135 |
| pubchem_noleak_n25000_s0 | vapor_pressure | 25000 | 2368 | 59 | 0.236 | 2.4493 |
| coconut_noleakskel_n25000_s2 | vapor_pressure | 25000 | 2368 | 58 | 0.232 | 2.3649 |
| coconut_n20000_s0 | vapor_pressure | 20000 | 2368 | 46 | 0.23 | 1.9426 |
| coconut_n20000_s0 | esol | 20000 | 1128 | 46 | 0.23 | 3.5874 |
| pubchem_noleak_n100000_s0 | vapor_pressure | 100000 | 2368 | 225 | 0.225 | 9.2483 |
| pubchem_n5000_s0 | bbbp | 5000 | 2050 | 11 | 0.22 | 0.5632 |
| coconut_noleak_n25000_s2 | ptp1b | 25000 | 460 | 52 | 0.208 | 8.945 |
| coconut_noleakskel_n25000_s0 | vapor_pressure | 25000 | 2368 | 52 | 0.208 | 2.1537 |
| coconut_noleak_n25000_s2 | bbbp | 25000 | 2050 | 50 | 0.2 | 2.3554 |
| coconut_noleakskel_n5000_s0 | esol | 5000 | 1128 | 10 | 0.2 | 0.8969 |
| pubchem_n5000_s1 | bbbp | 5000 | 2050 | 10 | 0.2 | 0.512 |
| pubchem_n1000_s0 | bbbp | 1000 | 2050 | 2 | 0.2 | 0.1024 |
| pubchem_n1000_s1 | vapor_pressure | 1000 | 2368 | 2 | 0.2 | 0.0845 |
| pubchem_n500_s0 | esol | 500 | 1128 | 1 | 0.2 | 0.0897 |
| pubchem_n500_s1 | vapor_pressure | 500 | 2368 | 1 | 0.2 | 0.0422 |
| pubchem_n500_s1 | esol | 500 | 1128 | 1 | 0.2 | 0.0897 |
| coconut_noleak_n25000_s2 | vapor_pressure | 25000 | 2368 | 49 | 0.196 | 1.9848 |
| coconut_noleak_n25000_s1 | vapor_pressure | 25000 | 2368 | 48 | 0.192 | 1.9426 |
| coconut_noleakskel_n25000_s0 | esol | 25000 | 1128 | 48 | 0.192 | 3.5874 |
| coconut_n20000_s0 | ptp1b | 20000 | 460 | 38 | 0.19 | 7.1101 |
| pubchem_n10000_s1 | esol | 10000 | 1128 | 19 | 0.19 | 1.6143 |
| pubchem_noleak_n25000_s1 | vapor_pressure | 25000 | 2368 | 47 | 0.188 | 1.9848 |
| pubchem_noleak_n25000_s2 | vapor_pressure | 25000 | 2368 | 47 | 0.188 | 1.9848 |
| coconut_noleak_n25000_s0 | bbbp | 25000 | 2050 | 46 | 0.184 | 2.2017 |
| coconut_noleak_n25000_s1 | bbbp | 25000 | 2050 | 45 | 0.18 | 1.9969 |
| coconut_noleakskel_n25000_s2 | esol | 25000 | 1128 | 44 | 0.176 | 3.6771 |
| coconut_noleakskel_n25000_s1 | esol | 25000 | 1128 | 42 | 0.168 | 3.4081 |
| coconut_noleak_n25000_s0 | vapor_pressure | 25000 | 2368 | 41 | 0.164 | 1.6892 |
| pubchem_noleak_n25000_s1 | bbbp | 25000 | 2050 | 41 | 0.164 | 1.9457 |
| coconut_noleak_n25000_s0 | ptp1b | 25000 | 460 | 40 | 0.16 | 8.2569 |
| pubchem_n100000_s0 | esol | 100000 | 1128 | 159 | 0.159 | 13.0942 |
| released_checkpoint_pool | esol | 944286 | 1128 | 1489 | 0.1577 | 93.4529 |
| pubchem_n100000_s1 | esol | 100000 | 1128 | 149 | 0.149 | 12.5561 |
| pubchem_n25000_s1 | esol | 25000 | 1128 | 37 | 0.148 | 3.2287 |
| pubchem_n50000_s1 | esol | 50000 | 1128 | 72 | 0.144 | 6.0987 |
| pubchem_n50000_s0 | esol | 50000 | 1128 | 70 | 0.14 | 6.1883 |
| zinc_nplike_noleak_n25000_s1 | np_origin | 25000 | 60170 | 35 | 0.14 | 0.0582 |
| coconut_noleakskel_n5000_s1 | esol | 5000 | 1128 | 7 | 0.14 | 0.6278 |
| coconut_noleak_n25000_s2 | esol | 25000 | 1128 | 34 | 0.136 | 2.3318 |
| pubchem_n25000_s0 | esol | 25000 | 1128 | 34 | 0.136 | 3.0493 |
| pubchem_n20000_s0 | esol | 20000 | 1128 | 27 | 0.135 | 2.4215 |
| pubchem_n10000_s0 | esol | 10000 | 1128 | 13 | 0.13 | 1.1659 |
| zinc_nplike_noleak_n25000_s2 | np_origin | 25000 | 60170 | 30 | 0.12 | 0.0499 |
| pubchem_n5000_s1 | esol | 5000 | 1128 | 6 | 0.12 | 0.5381 |
| coconut_noleak_n25000_s1 | esol | 25000 | 1128 | 29 | 0.116 | 1.9731 |
| zinc_nplike_n20000_s0 | np_origin | 20000 | 60170 | 23 | 0.115 | 0.0382 |
| zinc_nplike_noleak_n25000_s0 | np_origin | 25000 | 60170 | 27 | 0.108 | 0.0449 |
| pubchem_noleak_n100000_s0 | bbbp | 100000 | 2050 | 104 | 0.104 | 4.7619 |
| coconut_noleak_n25000_s0 | esol | 25000 | 1128 | 26 | 0.104 | 1.7937 |
| pubchem_n1000_s0 | esol | 1000 | 1128 | 1 | 0.1 | 0.0897 |
| pubchem_n1000_s1 | bbbp | 1000 | 2050 | 1 | 0.1 | 0.0512 |
| pubchem_n1000_s1 | esol | 1000 | 1128 | 1 | 0.1 | 0.0897 |
| pubchem_noleak_n25000_s2 | bbbp | 25000 | 2050 | 23 | 0.092 | 1.1777 |
| pubchem_noleak_n25000_s0 | bbbp | 25000 | 2050 | 22 | 0.088 | 1.1265 |
| coconut_noleakskel_n25000_s0 | ptp1b | 25000 | 460 | 21 | 0.084 | 4.8165 |
| pubchem_n5000_s0 | esol | 5000 | 1128 | 4 | 0.08 | 0.3587 |
| coconut_noleakskel_n25000_s2 | ptp1b | 25000 | 460 | 19 | 0.076 | 4.3578 |
| coconut_noleakskel_n5000_s0 | ptp1b | 5000 | 460 | 3 | 0.06 | 0.6881 |
| coconut_noleakskel_n25000_s1 | ptp1b | 25000 | 460 | 14 | 0.056 | 2.9817 |
| pubchem_noleak_n25000_s0 | esol | 25000 | 1128 | 14 | 0.056 | 1.1659 |
| pubchem_nearbench_noleak_n25000_s0 | ptp1b | 25000 | 460 | 13 | 0.052 | 2.2936 |
| pubchem_noleak_n100000_s0 | esol | 100000 | 1128 | 43 | 0.043 | 2.9596 |
| coconut_noleakskel_n5000_s1 | ptp1b | 5000 | 460 | 2 | 0.04 | 0.4587 |
| pubchem_n5000_s0 | ptp1b | 5000 | 460 | 2 | 0.04 | 0.4587 |
| pubchem_noleak_n25000_s1 | esol | 25000 | 1128 | 9 | 0.036 | 0.8072 |
| pubchem_noleak_n25000_s2 | esol | 25000 | 1128 | 9 | 0.036 | 0.7175 |
| pubchem_noleak_n25000_s2 | ptp1b | 25000 | 460 | 8 | 0.032 | 1.8349 |
| pubchem_n20000_s0 | ptp1b | 20000 | 460 | 6 | 0.03 | 1.3761 |
| pubchem_n10000_s0 | ptp1b | 10000 | 460 | 3 | 0.03 | 0.6881 |
| pubchem_nearbench_noleak_n25000_s1 | ptp1b | 25000 | 460 | 7 | 0.028 | 1.1468 |
| pubchem_nearbench_noleak_n25000_s2 | ptp1b | 25000 | 460 | 7 | 0.028 | 1.3761 |
| pubchem_n100000_s1 | ptp1b | 100000 | 460 | 26 | 0.026 | 5.5046 |
| pubchem_n25000_s0 | ptp1b | 25000 | 460 | 6 | 0.024 | 1.3761 |
| pubchem_n100000_s0 | ptp1b | 100000 | 460 | 23 | 0.023 | 5.0459 |
| pubchem_n50000_s0 | ptp1b | 50000 | 460 | 11 | 0.022 | 2.2936 |
| pubchem_n50000_s1 | ptp1b | 50000 | 460 | 11 | 0.022 | 2.5229 |
| released_checkpoint_pool | ptp1b | 944286 | 460 | 199 | 0.0211 | 26.8349 |
| pubchem_noleak_n100000_s0 | ptp1b | 100000 | 460 | 20 | 0.02 | 3.8991 |
| zinc_general_noleak_n25000_s1 | np_origin | 25000 | 60170 | 5 | 0.02 | 0.0083 |
| pubchem_n10000_s1 | ptp1b | 10000 | 460 | 2 | 0.02 | 0.4587 |
| pubchem_n25000_s1 | ptp1b | 25000 | 460 | 4 | 0.016 | 0.9174 |
| pubchem_noleak_n25000_s0 | ptp1b | 25000 | 460 | 4 | 0.016 | 0.9174 |
| pubchem_noleak_n25000_s1 | ptp1b | 25000 | 460 | 2 | 0.008 | 0.4587 |
| zinc_general_n20000_s0 | np_origin | 20000 | 60170 | 1 | 0.005 | 0.0017 |
| zinc_general_noleak_n25000_s0 | np_origin | 25000 | 60170 | 1 | 0.004 | 0.0017 |
| coconut_noleakskel_n25000_s0 | np_origin | 25000 | 60170 | 0 | 0.0 | 0.0 |
| coconut_noleakskel_n25000_s1 | np_origin | 25000 | 60170 | 0 | 0.0 | 0.0 |
| coconut_noleakskel_n25000_s2 | np_origin | 25000 | 60170 | 0 | 0.0 | 0.0 |
| coconut_noleakskel_n5000_s0 | np_origin | 5000 | 60170 | 0 | 0.0 | 0.0 |
| coconut_noleakskel_n5000_s1 | np_origin | 5000 | 60170 | 0 | 0.0 | 0.0 |
| pubchem_n1000_s0 | ptp1b | 1000 | 460 | 0 | 0.0 | 0.0 |
| pubchem_n1000_s1 | ptp1b | 1000 | 460 | 0 | 0.0 | 0.0 |
| pubchem_n5000_s1 | ptp1b | 5000 | 460 | 0 | 0.0 | 0.0 |
| pubchem_n500_s0 | ptp1b | 500 | 460 | 0 | 0.0 | 0.0 |
| pubchem_n500_s0 | bbbp | 500 | 2050 | 0 | 0.0 | 0.0 |
| pubchem_n500_s1 | ptp1b | 500 | 460 | 0 | 0.0 | 0.0 |
| pubchem_n500_s1 | bbbp | 500 | 2050 | 0 | 0.0 | 0.0 |
| zinc_general_n20000_s0 | vapor_pressure | 20000 | 2368 | 0 | 0.0 | 0.0 |
| zinc_general_n20000_s0 | ptp1b | 20000 | 460 | 0 | 0.0 | 0.0 |
| zinc_general_n20000_s0 | bbbp | 20000 | 2050 | 0 | 0.0 | 0.0 |
| zinc_general_n20000_s0 | esol | 20000 | 1128 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s0 | vapor_pressure | 25000 | 2368 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s0 | ptp1b | 25000 | 460 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s0 | bbbp | 25000 | 2050 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s0 | esol | 25000 | 1128 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s1 | vapor_pressure | 25000 | 2368 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s1 | ptp1b | 25000 | 460 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s1 | bbbp | 25000 | 2050 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s1 | esol | 25000 | 1128 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s2 | vapor_pressure | 25000 | 2368 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s2 | ptp1b | 25000 | 460 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s2 | np_origin | 25000 | 60170 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s2 | bbbp | 25000 | 2050 | 0 | 0.0 | 0.0 |
| zinc_general_noleak_n25000_s2 | esol | 25000 | 1128 | 0 | 0.0 | 0.0 |
| zinc_nplike_n20000_s0 | vapor_pressure | 20000 | 2368 | 0 | 0.0 | 0.0 |
| zinc_nplike_n20000_s0 | ptp1b | 20000 | 460 | 0 | 0.0 | 0.0 |
| zinc_nplike_n20000_s0 | bbbp | 20000 | 2050 | 0 | 0.0 | 0.0 |
| zinc_nplike_n20000_s0 | esol | 20000 | 1128 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s0 | vapor_pressure | 25000 | 2368 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s0 | ptp1b | 25000 | 460 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s0 | bbbp | 25000 | 2050 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s0 | esol | 25000 | 1128 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s1 | vapor_pressure | 25000 | 2368 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s1 | ptp1b | 25000 | 460 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s1 | bbbp | 25000 | 2050 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s1 | esol | 25000 | 1128 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s2 | vapor_pressure | 25000 | 2368 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s2 | ptp1b | 25000 | 460 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s2 | bbbp | 25000 | 2050 | 0 | 0.0 | 0.0 |
| zinc_nplike_noleak_n25000_s2 | esol | 25000 | 1128 | 0 | 0.0 | 0.0 |

## What the released-checkpoint row is and is not

`released_checkpoint_pool` is CheMeleon's published `cleaned_pubchem_1MM.smiles` (Zenodo 15733575), parsed and deduplicated to 944,286 molecules: the pretraining pool as distributed. It is not a verified account of what the released weights saw. We did not train that checkpoint, the reference pretrainer splits its SMILES 70/20/10 so roughly three in ten pool members were never trained on, and Amendment 15 records that we cannot verify the checkpoint came from this file in this order with this seed. Treat the row as an upper bound on the released checkpoint's exposure, and as the best statement available about it.
