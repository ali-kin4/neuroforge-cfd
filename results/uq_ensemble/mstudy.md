# Ensemble-size study (M = 2..5, all subsets of the 5 frozen members)

n_cases=200; sigma=ddof-0 member std; conformal alpha=0.1 split seed 0; error=rel-L2 speed of subset-mean. Gate (M=5 vs published Spearman 0.6103): PASS (|d|=0.0000).

| M | n_sub | sigma AUROC | fused AUROC | sigma rho | cov u | cov v | cov p | ECE p | q p |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 10 | 0.865 ± 0.038 | 0.910 ± 0.028 | 0.561 ± 0.040 | 0.894 | 0.903 | 0.905 | 0.199 | 9.98 |
| 3 | 10 | 0.877 ± 0.016 | 0.905 ± 0.014 | 0.613 ± 0.023 | 0.888 | 0.904 | 0.909 | 0.113 | 3.74 |
| 4 | 5 | 0.889 ± 0.016 | 0.912 ± 0.013 | 0.638 ± 0.017 | 0.884 | 0.906 | 0.913 | 0.087 | 2.75 |
| 5 | 1 | 0.894 ± 0.000 | 0.905 ± 0.000 | 0.654 ± 0.000 | 0.881 | 0.908 | 0.915 | 0.074 | 2.35 |
