# v3 prospective evaluation

Ground truth: two rare diseases become related on the date the same drug holds FDA or EMA orphan designations for both (distinct regulatory decisions). Each fold trains only on knowledge before its cutoff and is scored on relations established afterwards. Metrics are per query disease, means with bootstrap 95% CIs; candidates exclude already-related and nested (group/member) diseases.

## validation fold: cutoff 2014-01-01, until 2018-01-01

818 relations known at the cutoff, 532 new relations to predict, 303 query diseases, 511 diseases with a designation before the cutoff. Stacker trained on 103274 pairs (2834 positive) from origins 2006, 2008, 2010, 2012.

### All 9,525 nodes as candidates (`full`)

| model | queries | MAP | AUC | MRR | Hits@10 | nDCG@10 | Recall@50 |
|---|---|---|---|---|---|---|---|
| random | 303 | 0.001 [0.001, 0.002] | 0.523 | 0.003 | 0.007 | 0.001 | 0.005 |
| degree | 303 | 0.065 [0.052, 0.081] | 0.870 | 0.116 | 0.300 | 0.080 | 0.289 |
| adamic_adar | 303 | 0.062 [0.046, 0.080] | 0.618 | 0.116 | 0.228 | 0.081 | 0.223 |
| drug_mechanism | 303 | 0.037 [0.025, 0.051] | 0.668 | 0.067 | 0.125 | 0.043 | 0.129 |
| v1_weighted_jaccard | 303 | 0.014 [0.009, 0.022] | 0.778 | 0.029 | 0.063 | 0.015 | 0.079 |
| v2_retrained | 303 | 0.063 [0.046, 0.080] | 0.863 | 0.125 | 0.241 | 0.080 | 0.251 |
| gbm_static | 303 | 0.059 [0.045, 0.074] | 0.861 | 0.119 | 0.241 | 0.075 | 0.251 |
| v3_neural | 303 | 0.056 [0.044, 0.070] | 0.870 | 0.116 | 0.254 | 0.076 | 0.252 |
| v3_static | 303 | 0.059 [0.045, 0.074] | 0.880 | 0.120 | 0.251 | 0.077 | 0.260 |
| v3_stacker | 303 | 0.089 [0.072, 0.108] | 0.932 | 0.159 | 0.327 | 0.115 | 0.368 |
| stacker_no_static | 303 | 0.042 [0.033, 0.052] | 0.794 | 0.094 | 0.221 | 0.051 | 0.221 |
| stacker_no_graph | 303 | 0.078 [0.064, 0.094] | 0.925 | 0.149 | 0.317 | 0.104 | 0.343 |
| stacker_no_history | 303 | 0.086 [0.070, 0.105] | 0.915 | 0.158 | 0.353 | 0.113 | 0.344 |
| stacker_no_mechanism | 303 | 0.085 [0.069, 0.103] | 0.931 | 0.148 | 0.320 | 0.105 | 0.360 |
| stacker_static_only | 303 | 0.080 [0.065, 0.097] | 0.908 | 0.157 | 0.314 | 0.104 | 0.309 |

### Only diseases already designated before the cutoff as candidates (`warm`)

| model | queries | MAP | AUC | MRR | Hits@10 | nDCG@10 | Recall@50 |
|---|---|---|---|---|---|---|---|
| random | 268 | 0.022 [0.015, 0.032] | 0.530 | 0.037 | 0.078 | 0.018 | 0.103 |
| degree | 268 | 0.082 [0.066, 0.100] | 0.754 | 0.132 | 0.340 | 0.097 | 0.371 |
| adamic_adar | 268 | 0.087 [0.069, 0.106] | 0.633 | 0.138 | 0.261 | 0.099 | 0.353 |
| drug_mechanism | 268 | 0.062 [0.046, 0.081] | 0.632 | 0.097 | 0.172 | 0.062 | 0.270 |
| v1_weighted_jaccard | 268 | 0.075 [0.058, 0.093] | 0.720 | 0.119 | 0.250 | 0.084 | 0.324 |
| v2_retrained | 268 | 0.141 [0.116, 0.168] | 0.761 | 0.243 | 0.440 | 0.178 | 0.447 |
| gbm_static | 268 | 0.150 [0.122, 0.178] | 0.768 | 0.234 | 0.418 | 0.181 | 0.483 |
| v3_neural | 268 | 0.136 [0.114, 0.160] | 0.790 | 0.224 | 0.444 | 0.171 | 0.482 |
| v3_static | 268 | 0.145 [0.120, 0.171] | 0.792 | 0.242 | 0.444 | 0.179 | 0.472 |
| v3_stacker | 268 | 0.151 [0.125, 0.180] | 0.805 | 0.233 | 0.440 | 0.182 | 0.504 |
| stacker_no_static | 268 | 0.073 [0.059, 0.088] | 0.675 | 0.135 | 0.310 | 0.086 | 0.377 |
| stacker_no_graph | 268 | 0.141 [0.116, 0.168] | 0.784 | 0.224 | 0.414 | 0.169 | 0.491 |
| stacker_no_history | 268 | 0.146 [0.122, 0.173] | 0.809 | 0.242 | 0.455 | 0.179 | 0.522 |
| stacker_no_mechanism | 268 | 0.144 [0.119, 0.173] | 0.806 | 0.218 | 0.407 | 0.169 | 0.514 |
| stacker_static_only | 268 | 0.136 [0.113, 0.161] | 0.795 | 0.226 | 0.444 | 0.168 | 0.471 |

### All nodes; positives restricted to relations where both designations led to approval (`full_approved`)

| model | queries | MAP | AUC | MRR | Hits@10 | nDCG@10 | Recall@50 |
|---|---|---|---|---|---|---|---|
| random | 68 | 0.002 [0.000, 0.006] | 0.484 | 0.004 | 0.015 | 0.004 | 0.007 |
| degree | 68 | 0.055 [0.029, 0.091] | 0.877 | 0.071 | 0.235 | 0.059 | 0.319 |
| adamic_adar | 68 | 0.098 [0.061, 0.141] | 0.683 | 0.186 | 0.294 | 0.127 | 0.342 |
| drug_mechanism | 68 | 0.041 [0.021, 0.064] | 0.704 | 0.069 | 0.162 | 0.054 | 0.156 |
| v1_weighted_jaccard | 68 | 0.019 [0.010, 0.032] | 0.884 | 0.031 | 0.074 | 0.019 | 0.106 |
| v2_retrained | 68 | 0.139 [0.089, 0.194] | 0.939 | 0.233 | 0.382 | 0.166 | 0.415 |
| gbm_static | 68 | 0.129 [0.087, 0.178] | 0.936 | 0.232 | 0.426 | 0.163 | 0.441 |
| v3_neural | 68 | 0.115 [0.079, 0.155] | 0.958 | 0.221 | 0.412 | 0.159 | 0.420 |
| v3_static | 68 | 0.137 [0.093, 0.186] | 0.964 | 0.236 | 0.412 | 0.176 | 0.454 |
| v3_stacker | 68 | 0.156 [0.105, 0.213] | 0.984 | 0.244 | 0.412 | 0.192 | 0.514 |
| stacker_no_static | 68 | 0.047 [0.031, 0.066] | 0.890 | 0.064 | 0.176 | 0.043 | 0.316 |
| stacker_no_graph | 68 | 0.146 [0.099, 0.199] | 0.982 | 0.235 | 0.441 | 0.185 | 0.548 |
| stacker_no_history | 68 | 0.156 [0.110, 0.204] | 0.976 | 0.237 | 0.529 | 0.213 | 0.522 |
| stacker_no_mechanism | 68 | 0.152 [0.102, 0.210] | 0.985 | 0.234 | 0.441 | 0.185 | 0.504 |
| stacker_static_only | 68 | 0.175 [0.117, 0.238] | 0.976 | 0.270 | 0.471 | 0.211 | 0.470 |

### Paired comparisons (full gallery, per-query differences)

| comparison | metric | mean difference [95% CI] | Wilcoxon p (greater) |
|---|---|---|---|
| v3_stacker vs random | map | +0.0875 [+0.0702, +0.1050] | 1.3e-49 |
| v3_stacker vs random | auc | +0.4090 [+0.3837, +0.4347] | 3.7e-50 |
| v3_stacker vs degree | map | +0.0237 [+0.0010, +0.0468] | 0.047 |
| v3_stacker vs degree | auc | +0.0621 [+0.0413, +0.0847] | 1.8e-08 |
| v3_stacker vs adamic_adar | map | +0.0268 [+0.0101, +0.0452] | 7.5e-11 |
| v3_stacker vs adamic_adar | auc | +0.3143 [+0.2926, +0.3353] | 1.9e-47 |
| v3_stacker vs drug_mechanism | map | +0.0521 [+0.0301, +0.0739] | 2e-21 |
| v3_stacker vs drug_mechanism | auc | +0.2635 [+0.2395, +0.2877] | 6.4e-43 |
| v3_stacker vs v1_weighted_jaccard | map | +0.0746 [+0.0587, +0.0917] | 3.5e-43 |
| v3_stacker vs v1_weighted_jaccard | auc | +0.1542 [+0.1323, +0.1757] | 4.8e-39 |
| v3_stacker vs v2_retrained | map | +0.0261 [+0.0127, +0.0398] | 8.8e-18 |
| v3_stacker vs v2_retrained | auc | +0.0688 [+0.0548, +0.0846] | 5.1e-23 |
| v3_stacker vs gbm_static | map | +0.0299 [+0.0193, +0.0409] | 1.5e-18 |
| v3_stacker vs gbm_static | auc | +0.0705 [+0.0541, +0.0889] | 4.3e-23 |
| v3_stacker vs v3_neural | map | +0.0326 [+0.0213, +0.0445] | 2e-17 |
| v3_stacker vs v3_neural | auc | +0.0619 [+0.0475, +0.0781] | 6.2e-22 |
| v3_stacker vs v3_static | map | +0.0295 [+0.0178, +0.0419] | 6.2e-17 |
| v3_stacker vs v3_static | auc | +0.0517 [+0.0390, +0.0652] | 4.2e-20 |
| v3_stacker vs stacker_no_static | map | +0.0472 [+0.0303, +0.0645] | 9.1e-16 |
| v3_stacker vs stacker_no_static | auc | +0.1380 [+0.1127, +0.1654] | 1.2e-30 |
| v3_stacker vs stacker_no_graph | map | +0.0107 [+0.0028, +0.0200] | 0.0087 |
| v3_stacker vs stacker_no_graph | auc | +0.0069 [+0.0014, +0.0124] | 1.1e-05 |
| v3_stacker vs stacker_no_history | map | +0.0031 [-0.0064, +0.0135] | 0.12 |
| v3_stacker vs stacker_no_history | auc | +0.0172 [+0.0096, +0.0252] | 1.1e-08 |
| v3_stacker vs stacker_no_mechanism | map | +0.0042 [-0.0000, +0.0085] | 0.11 |
| v3_stacker vs stacker_no_mechanism | auc | +0.0007 [-0.0040, +0.0054] | 0.41 |
| v3_stacker vs stacker_static_only | map | +0.0093 [-0.0045, +0.0233] | 1.8e-05 |
| v3_stacker vs stacker_static_only | auc | +0.0238 [+0.0151, +0.0326] | 1.7e-15 |
| v3_static vs v1_weighted_jaccard | map | +0.0451 [+0.0322, +0.0597] | 1.5e-39 |
| v3_static vs v1_weighted_jaccard | auc | +0.1025 [+0.0825, +0.1207] | 1.1e-34 |
| v3_static vs v2_retrained | map | -0.0034 [-0.0108, +0.0024] | 6.4e-08 |
| v3_static vs v2_retrained | auc | +0.0171 [+0.0080, +0.0260] | 9e-08 |
| v3_static vs gbm_static | map | +0.0003 [-0.0089, +0.0092] | 0.02 |
| v3_static vs gbm_static | auc | +0.0188 [+0.0059, +0.0314] | 0.00049 |
| v3_static vs v3_neural | map | +0.0031 [-0.0021, +0.0095] | 0.19 |
| v3_static vs v3_neural | auc | +0.0103 [+0.0029, +0.0173] | 7.1e-06 |

### MAP by query stratum (full gallery)

| stratum | queries | random | degree | adamic_adar | drug_mechanism | v1_weighted_jaccard | v2_retrained | gbm_static | v3_neural | v3_static | v3_stacker | stacker_no_static | stacker_no_graph | stacker_no_history | stacker_no_mechanism | stacker_static_only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm query | 235 | 0.002 | 0.059 | 0.080 | 0.041 | 0.010 | 0.051 | 0.050 | 0.048 | 0.048 | 0.085 | 0.049 | 0.071 | 0.083 | 0.081 | 0.074 |
| cold query | 68 | 0.001 | 0.085 | 0.001 | 0.020 | 0.029 | 0.105 | 0.089 | 0.083 | 0.098 | 0.103 | 0.018 | 0.102 | 0.095 | 0.099 | 0.099 |
| group query | 76 | 0.001 | 0.046 | 0.054 | 0.027 | 0.009 | 0.041 | 0.032 | 0.040 | 0.039 | 0.064 | 0.039 | 0.058 | 0.060 | 0.062 | 0.060 |
| disease query | 227 | 0.001 | 0.071 | 0.065 | 0.040 | 0.016 | 0.070 | 0.068 | 0.062 | 0.066 | 0.097 | 0.042 | 0.085 | 0.094 | 0.092 | 0.086 |
| oncology query | 73 | 0.001 | 0.107 | 0.138 | 0.036 | 0.012 | 0.079 | 0.073 | 0.088 | 0.084 | 0.121 | 0.069 | 0.113 | 0.127 | 0.117 | 0.123 |
| non-oncology query | 230 | 0.001 | 0.052 | 0.038 | 0.037 | 0.015 | 0.058 | 0.055 | 0.046 | 0.051 | 0.079 | 0.033 | 0.067 | 0.072 | 0.074 | 0.066 |

### Global ranking of new pairs among designated diseases (base rate 0.0025: 325 of 129123 pairs)

| model | precision@25 | precision@100 | precision@500 |
|---|---|---|---|
| random | 0.000 | 0.000 | 0.004 |
| degree | 0.000 | 0.020 | 0.016 |
| adamic_adar | 0.240 | 0.170 | 0.118 |
| drug_mechanism | 0.000 | 0.010 | 0.024 |
| v1_weighted_jaccard | 0.000 | 0.010 | 0.014 |
| v2_retrained | 0.040 | 0.060 | 0.062 |
| gbm_static | 0.040 | 0.100 | 0.062 |
| v3_neural | 0.120 | 0.090 | 0.080 |
| v3_static | 0.120 | 0.080 | 0.068 |
| v3_stacker | 0.280 | 0.130 | 0.094 |
| stacker_no_static | 0.040 | 0.070 | 0.040 |
| stacker_no_graph | 0.200 | 0.130 | 0.088 |
| stacker_no_history | 0.160 | 0.160 | 0.106 |
| stacker_no_mechanism | 0.200 | 0.140 | 0.096 |
| stacker_static_only | 0.280 | 0.150 | 0.080 |

### v3 stacker: permutation importance (drop in pooled AUC on training pairs)

| feature family | AUC drop |
|---|---|
| static | 0.2722 |
| history | 0.0241 |
| graph | 0.0011 |
| node | 0.0005 |
| mechanism | 0.0001 |

## test fold: cutoff 2018-01-01, until end of data

1350 relations known at the cutoff, 1123 new relations to predict, 447 query diseases, 638 diseases with a designation before the cutoff. Stacker trained on 150666 pairs (4746 positive) from origins 2010, 2012, 2014, 2016.

### All 9,525 nodes as candidates (`full`)

| model | queries | MAP | AUC | MRR | Hits@10 | nDCG@10 | Recall@50 |
|---|---|---|---|---|---|---|---|
| random | 447 | 0.001 [0.001, 0.001] | 0.497 | 0.002 | 0.000 | 0.000 | 0.003 |
| degree | 447 | 0.070 [0.060, 0.080] | 0.899 | 0.152 | 0.385 | 0.093 | 0.295 |
| adamic_adar | 447 | 0.050 [0.040, 0.062] | 0.597 | 0.118 | 0.251 | 0.074 | 0.165 |
| drug_mechanism | 447 | 0.038 [0.027, 0.049] | 0.674 | 0.074 | 0.159 | 0.042 | 0.136 |
| v1_weighted_jaccard | 447 | 0.014 [0.011, 0.018] | 0.767 | 0.034 | 0.067 | 0.014 | 0.089 |
| v2_retrained | 447 | 0.061 [0.049, 0.073] | 0.841 | 0.135 | 0.277 | 0.083 | 0.251 |
| gbm_static | 447 | 0.055 [0.045, 0.067] | 0.854 | 0.117 | 0.226 | 0.069 | 0.244 |
| v3_neural | 447 | 0.063 [0.052, 0.075] | 0.871 | 0.140 | 0.284 | 0.080 | 0.264 |
| v3_static | 447 | 0.064 [0.053, 0.077] | 0.868 | 0.144 | 0.293 | 0.086 | 0.262 |
| v3_stacker | 447 | 0.113 [0.096, 0.130] | 0.935 | 0.235 | 0.418 | 0.140 | 0.373 |
| stacker_no_static | 447 | 0.072 [0.061, 0.085] | 0.868 | 0.184 | 0.347 | 0.101 | 0.271 |
| stacker_no_graph | 447 | 0.096 [0.082, 0.111] | 0.935 | 0.214 | 0.385 | 0.126 | 0.327 |
| stacker_no_history | 447 | 0.103 [0.088, 0.119] | 0.926 | 0.212 | 0.403 | 0.133 | 0.356 |
| stacker_no_mechanism | 447 | 0.111 [0.094, 0.128] | 0.938 | 0.225 | 0.427 | 0.140 | 0.394 |
| stacker_static_only | 447 | 0.086 [0.073, 0.101] | 0.919 | 0.178 | 0.383 | 0.111 | 0.336 |

### Only diseases already designated before the cutoff as candidates (`warm`)

| model | queries | MAP | AUC | MRR | Hits@10 | nDCG@10 | Recall@50 |
|---|---|---|---|---|---|---|---|
| random | 417 | 0.014 [0.013, 0.016] | 0.497 | 0.024 | 0.046 | 0.007 | 0.079 |
| degree | 417 | 0.080 [0.069, 0.091] | 0.732 | 0.162 | 0.412 | 0.103 | 0.342 |
| adamic_adar | 417 | 0.069 [0.057, 0.083] | 0.595 | 0.139 | 0.288 | 0.087 | 0.231 |
| drug_mechanism | 417 | 0.056 [0.045, 0.069] | 0.626 | 0.096 | 0.221 | 0.057 | 0.222 |
| v1_weighted_jaccard | 417 | 0.087 [0.074, 0.102] | 0.725 | 0.161 | 0.343 | 0.106 | 0.317 |
| v2_retrained | 417 | 0.142 [0.121, 0.164] | 0.750 | 0.253 | 0.463 | 0.174 | 0.416 |
| gbm_static | 417 | 0.133 [0.115, 0.153] | 0.767 | 0.225 | 0.463 | 0.161 | 0.456 |
| v3_neural | 417 | 0.149 [0.129, 0.171] | 0.794 | 0.267 | 0.504 | 0.186 | 0.455 |
| v3_static | 417 | 0.147 [0.127, 0.169] | 0.783 | 0.262 | 0.494 | 0.181 | 0.449 |
| v3_stacker | 417 | 0.156 [0.135, 0.178] | 0.810 | 0.288 | 0.513 | 0.192 | 0.490 |
| stacker_no_static | 417 | 0.092 [0.079, 0.106] | 0.718 | 0.201 | 0.388 | 0.119 | 0.368 |
| stacker_no_graph | 417 | 0.139 [0.120, 0.158] | 0.787 | 0.267 | 0.475 | 0.175 | 0.423 |
| stacker_no_history | 417 | 0.143 [0.124, 0.161] | 0.822 | 0.260 | 0.494 | 0.175 | 0.497 |
| stacker_no_mechanism | 417 | 0.156 [0.135, 0.177] | 0.814 | 0.279 | 0.520 | 0.192 | 0.493 |
| stacker_static_only | 417 | 0.130 [0.113, 0.149] | 0.812 | 0.236 | 0.484 | 0.162 | 0.476 |

### All nodes; positives restricted to relations where both designations led to approval (`full_approved`)

| model | queries | MAP | AUC | MRR | Hits@10 | nDCG@10 | Recall@50 |
|---|---|---|---|---|---|---|---|
| random | 46 | 0.000 [0.000, 0.000] | 0.432 | 0.000 | 0.000 | 0.000 | 0.000 |
| degree | 46 | 0.030 [0.011, 0.059] | 0.839 | 0.043 | 0.130 | 0.039 | 0.181 |
| adamic_adar | 46 | 0.063 [0.019, 0.122] | 0.596 | 0.061 | 0.174 | 0.088 | 0.196 |
| drug_mechanism | 46 | 0.046 [0.009, 0.102] | 0.736 | 0.046 | 0.065 | 0.045 | 0.174 |
| v1_weighted_jaccard | 46 | 0.008 [0.004, 0.013] | 0.851 | 0.007 | 0.000 | 0.000 | 0.087 |
| v2_retrained | 46 | 0.056 [0.035, 0.080] | 0.922 | 0.060 | 0.283 | 0.093 | 0.453 |
| gbm_static | 46 | 0.086 [0.042, 0.146] | 0.933 | 0.088 | 0.196 | 0.095 | 0.431 |
| v3_neural | 46 | 0.056 [0.035, 0.084] | 0.915 | 0.061 | 0.239 | 0.082 | 0.457 |
| v3_static | 46 | 0.060 [0.038, 0.087] | 0.928 | 0.065 | 0.304 | 0.099 | 0.475 |
| v3_stacker | 46 | 0.184 [0.099, 0.284] | 0.975 | 0.200 | 0.326 | 0.203 | 0.475 |
| stacker_no_static | 46 | 0.035 [0.016, 0.062] | 0.850 | 0.047 | 0.109 | 0.042 | 0.330 |
| stacker_no_graph | 46 | 0.136 [0.066, 0.223] | 0.963 | 0.160 | 0.283 | 0.156 | 0.457 |
| stacker_no_history | 46 | 0.146 [0.070, 0.234] | 0.965 | 0.161 | 0.261 | 0.170 | 0.395 |
| stacker_no_mechanism | 46 | 0.155 [0.082, 0.239] | 0.975 | 0.173 | 0.326 | 0.184 | 0.540 |
| stacker_static_only | 46 | 0.088 [0.044, 0.145] | 0.961 | 0.100 | 0.239 | 0.112 | 0.406 |

### Paired comparisons (full gallery, per-query differences)

| comparison | metric | mean difference [95% CI] | Wilcoxon p (greater) |
|---|---|---|---|
| v3_stacker vs random | map | +0.1117 [+0.0953, +0.1296] | 8.9e-74 |
| v3_stacker vs random | auc | +0.4382 [+0.4164, +0.4592] | 4.7e-72 |
| v3_stacker vs degree | map | +0.0431 [+0.0236, +0.0634] | 6.8e-05 |
| v3_stacker vs degree | auc | +0.0357 [+0.0201, +0.0518] | 2.9e-08 |
| v3_stacker vs adamic_adar | map | +0.0627 [+0.0453, +0.0809] | 3e-36 |
| v3_stacker vs adamic_adar | auc | +0.3379 [+0.3216, +0.3539] | 3.6e-71 |
| v3_stacker vs drug_mechanism | map | +0.0752 [+0.0553, +0.0953] | 2.3e-38 |
| v3_stacker vs drug_mechanism | auc | +0.2609 [+0.2395, +0.2807] | 2e-62 |
| v3_stacker vs v1_weighted_jaccard | map | +0.0985 [+0.0827, +0.1156] | 7.3e-63 |
| v3_stacker vs v1_weighted_jaccard | auc | +0.1676 [+0.1515, +0.1830] | 3.2e-63 |
| v3_stacker vs v2_retrained | map | +0.0521 [+0.0388, +0.0665] | 6.1e-26 |
| v3_stacker vs v2_retrained | auc | +0.0939 [+0.0807, +0.1077] | 2.3e-44 |
| v3_stacker vs gbm_static | map | +0.0575 [+0.0440, +0.0720] | 3.1e-32 |
| v3_stacker vs gbm_static | auc | +0.0809 [+0.0687, +0.0932] | 4.5e-40 |
| v3_stacker vs v3_neural | map | +0.0499 [+0.0362, +0.0649] | 2.3e-24 |
| v3_stacker vs v3_neural | auc | +0.0643 [+0.0523, +0.0771] | 6.5e-30 |
| v3_stacker vs v3_static | map | +0.0484 [+0.0350, +0.0627] | 2e-22 |
| v3_stacker vs v3_static | auc | +0.0672 [+0.0552, +0.0804] | 8.8e-34 |
| v3_stacker vs stacker_no_static | map | +0.0405 [+0.0255, +0.0571] | 2.9e-12 |
| v3_stacker vs stacker_no_static | auc | +0.0664 [+0.0513, +0.0816] | 1.1e-26 |
| v3_stacker vs stacker_no_graph | map | +0.0172 [+0.0080, +0.0267] | 1.8e-05 |
| v3_stacker vs stacker_no_graph | auc | +0.0003 [-0.0035, +0.0043] | 0.034 |
| v3_stacker vs stacker_no_history | map | +0.0097 [-0.0010, +0.0213] | 0.016 |
| v3_stacker vs stacker_no_history | auc | +0.0089 [+0.0030, +0.0147] | 5.4e-06 |
| v3_stacker vs stacker_no_mechanism | map | +0.0021 [-0.0050, +0.0088] | 0.87 |
| v3_stacker vs stacker_no_mechanism | auc | -0.0034 [-0.0056, -0.0014] | 0.99 |
| v3_stacker vs stacker_static_only | map | +0.0267 [+0.0134, +0.0402] | 2.4e-08 |
| v3_stacker vs stacker_static_only | auc | +0.0161 [+0.0096, +0.0229] | 5.5e-14 |
| v3_static vs v1_weighted_jaccard | map | +0.0501 [+0.0394, +0.0615] | 1.6e-57 |
| v3_static vs v1_weighted_jaccard | auc | +0.1004 [+0.0873, +0.1128] | 7.8e-53 |
| v3_static vs v2_retrained | map | +0.0036 [+0.0013, +0.0059] | 1.5e-17 |
| v3_static vs v2_retrained | auc | +0.0267 [+0.0210, +0.0327] | 3.9e-25 |
| v3_static vs gbm_static | map | +0.0091 [+0.0012, +0.0170] | 2.9e-07 |
| v3_static vs gbm_static | auc | +0.0137 [+0.0057, +0.0220] | 1.2e-05 |
| v3_static vs v3_neural | map | +0.0014 [-0.0017, +0.0046] | 0.92 |
| v3_static vs v3_neural | auc | -0.0029 [-0.0075, +0.0019] | 0.76 |

### MAP by query stratum (full gallery)

| stratum | queries | random | degree | adamic_adar | drug_mechanism | v1_weighted_jaccard | v2_retrained | gbm_static | v3_neural | v3_static | v3_stacker | stacker_no_static | stacker_no_graph | stacker_no_history | stacker_no_mechanism | stacker_static_only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| warm query | 353 | 0.001 | 0.074 | 0.063 | 0.042 | 0.013 | 0.055 | 0.051 | 0.057 | 0.059 | 0.120 | 0.081 | 0.103 | 0.109 | 0.120 | 0.085 |
| cold query | 94 | 0.001 | 0.053 | 0.001 | 0.019 | 0.018 | 0.082 | 0.070 | 0.086 | 0.086 | 0.084 | 0.039 | 0.067 | 0.080 | 0.076 | 0.091 |
| group query | 94 | 0.001 | 0.077 | 0.065 | 0.048 | 0.010 | 0.037 | 0.036 | 0.037 | 0.038 | 0.108 | 0.079 | 0.092 | 0.096 | 0.104 | 0.063 |
| disease query | 353 | 0.001 | 0.068 | 0.046 | 0.035 | 0.015 | 0.067 | 0.060 | 0.070 | 0.071 | 0.114 | 0.070 | 0.097 | 0.105 | 0.112 | 0.092 |
| oncology query | 83 | 0.001 | 0.134 | 0.098 | 0.049 | 0.012 | 0.036 | 0.045 | 0.046 | 0.042 | 0.153 | 0.142 | 0.138 | 0.154 | 0.145 | 0.113 |
| non-oncology query | 364 | 0.001 | 0.055 | 0.039 | 0.035 | 0.015 | 0.066 | 0.058 | 0.067 | 0.069 | 0.103 | 0.056 | 0.086 | 0.091 | 0.103 | 0.080 |

### Global ranking of new pairs among designated diseases (base rate 0.0042: 840 of 201410 pairs)

| model | precision@25 | precision@100 | precision@500 |
|---|---|---|---|
| random | 0.000 | 0.000 | 0.000 |
| degree | 0.080 | 0.040 | 0.030 |
| adamic_adar | 0.280 | 0.260 | 0.134 |
| drug_mechanism | 0.000 | 0.010 | 0.032 |
| v1_weighted_jaccard | 0.000 | 0.020 | 0.018 |
| v2_retrained | 0.080 | 0.070 | 0.074 |
| gbm_static | 0.040 | 0.070 | 0.076 |
| v3_neural | 0.080 | 0.090 | 0.092 |
| v3_static | 0.120 | 0.080 | 0.084 |
| v3_stacker | 0.320 | 0.200 | 0.168 |
| stacker_no_static | 0.400 | 0.260 | 0.146 |
| stacker_no_graph | 0.280 | 0.240 | 0.148 |
| stacker_no_history | 0.280 | 0.240 | 0.148 |
| stacker_no_mechanism | 0.200 | 0.220 | 0.176 |
| stacker_static_only | 0.200 | 0.200 | 0.142 |

### v3 stacker: permutation importance (drop in pooled AUC on training pairs)

| feature family | AUC drop |
|---|---|
| static | 0.2769 |
| history | 0.0371 |
| graph | 0.0050 |
| node | 0.0013 |
| mechanism | 0.0003 |

Production stacker variant (highest full-gallery MAP on the validation fold): `v3_stacker`
