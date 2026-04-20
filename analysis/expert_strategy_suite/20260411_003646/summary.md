# Expert Strategy Summary (landed_z=0.01, target_min_z=0.2)

All runs use Isaac Sim 6-DoF, pred_mode=noise, v_prey=1.5, 512 parallel envs, 1024 episodes.

| strategy | episodes | capture | goal | landed | timeout | any coll | drone coll | first_capture_step_mean | log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_best | 1024 | 77.7% | 8.8% | 0.0% | 13.5% | 65.1% | 28.7% | 355.2 | /data/uavlab/multi-uav-pursuit2/analysis/expert_strategy_suite/20260411_003646/logs/baseline_best.log |
| expert2_both_on | 1024 | 74.7% | 8.2% | 0.0% | 17.1% | 61.3% | 26.6% | 300.5 | /data/uavlab/multi-uav-pursuit2/analysis/expert_strategy_suite/20260411_003646/logs/expert2_both_on.log |
| expert2_none | 1024 | 72.9% | 10.1% | 0.0% | 17.0% | 60.4% | 16.5% | 310.3 | /data/uavlab/multi-uav-pursuit2/analysis/expert_strategy_suite/20260411_003646/logs/expert2_none.log |
| expert2_close_only | 1024 | 75.9% | 8.8% | 0.0% | 15.3% | 61.1% | 24.8% | 307.2 | /data/uavlab/multi-uav-pursuit2/analysis/expert_strategy_suite/20260411_003646/logs/expert2_close_only.log |
| expert2_rush_only | 1024 | 72.9% | 10.3% | 0.0% | 16.8% | 60.4% | 16.7% | 312.4 | /data/uavlab/multi-uav-pursuit2/analysis/expert_strategy_suite/20260411_003646/logs/expert2_rush_only.log |
| expert2_goal_off_close_only | 1024 | 77.4% | 7.5% | 0.0% | 15.0% | 61.1% | 24.8% | 325.2 | /data/uavlab/multi-uav-pursuit2/analysis/expert_strategy_suite/20260411_003646/logs/expert2_goal_off_close_only.log |
| expert2_goal_off_close_only_staggered | 1024 | 73.6% | 7.2% | 0.0% | 19.1% | 63.9% | 32.5% | 303.7 | /data/uavlab/multi-uav-pursuit2/analysis/expert_strategy_suite/20260411_003646/logs/expert2_goal_off_close_only_staggered.log |
