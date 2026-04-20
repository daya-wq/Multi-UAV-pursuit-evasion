from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

path = glob.glob('/data/uavlab/multi-uav-pursuit2/runs/HideAndSeek_20260326_002011/events.out.tfevents.*')[0]
ea = EventAccumulator(path)
ea.Reload()

tags_to_check = [
    'train/stats.success',
    'train/stats.collision_reward',
    'train/stats.distance_reward',
    'train/stats.speed_reward',
    'train/stats.smoothness_reward',
    'train/stats.catch_reward',
    'drone/action_norm',
    'drone/cmd_norm',
    'drone/entropy',
    'drone/actor_log_std_mean',
    'drone/actor_log_std_max',
    'drone/actor_grad_norm',
    'drone/ESS',
]

print(f"Stats from {path}")
for tag in tags_to_check:
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        if len(events) > 0:
            last_val = events[-1].value
            max_val = max([e.value for e in events])
            min_val = min([e.value for e in events])
            print(f"{tag}: current={last_val:.4f}, min={min_val:.4f}, max={max_val:.4f} (len={len(events)})")
    else:
        print(f"{tag}: NOT FOUND")
