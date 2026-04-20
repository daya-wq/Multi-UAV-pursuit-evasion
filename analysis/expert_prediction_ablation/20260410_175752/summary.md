# Expert Prediction Ablation

- `v_prey=1.5`
- `v_drone=1.5`
- `episode_length=1000`
- `generic_batch_envs=512`
- `num_waves=1`

| requested | actual | episodes | capture | goal | landed | timeout | cap_steps_mean | pred_pos_err | pred_next_err | delta_vs_noise | log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| noise | noise | 512 | 72.07% | 8.01% | 16.02% | 3.91% | 385.1 | 0.2504 | 0.2499 | +0.00 pp | [log](noise.log) |
| tp_net | tp_net | 512 | 64.65% | 13.28% | 16.60% | 5.47% | 352.1 | 0.3568 | 0.3581 | -7.42 pp | [log](tp_net.log) |
| oracle_pos | oracle_pos | 512 | 75.20% | 7.23% | 14.45% | 3.12% | 376.3 | 0.0000 | 0.0113 | +3.12 pp | [log](oracle_pos.log) |
| oracle_next | oracle_next | 512 | 76.17% | 7.03% | 13.87% | 2.93% | 378.2 | 0.0113 | 0.0000 | +4.10 pp | [log](oracle_next.log) |
