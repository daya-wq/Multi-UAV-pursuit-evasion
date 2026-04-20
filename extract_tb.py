import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = "/data/uavlab/multi-uav-pursuit2/runs/HideAndSeek_20260326_002011/events.out.tfevents.1774455611.uavlab.1257216.0"
ea = EventAccumulator(path)
ea.Reload()

metrics_to_check = [
    "train/stats.success", "train/stats.catch_reward", "train/stats.first_capture_step", "train/stats.return",
    "train/stats.terminal_reward", "train/stats.goal_progress_reward", "train/stats.capture_progress_reward",
    "train/stats.coop_reward", "train/stats.phi_team", "train/stats.d_i_mean",
    "train/stats.goal_reached", "train/stats.target_predicted_error",
    "train/stats.collision_drone", "train/stats.pursuer_collisions_count", "train/stats.collision_floor",
    "train/stats.any_landed", "train/stats.landed_penalty", "train/stats.smoothness_reward",
    "drone/action_norm", "drone/cmd_norm", "drone/entropy", "drone/ESS", "drone/actor_grad_norm",
    "drone/value_loss", "drone/policy_loss"
]

print("Metrics Summary:")
for tag in ea.Tags()['scalars']:
    if tag in metrics_to_check or any(m in tag for m in ['success', 'reward', 'loss', 'entropy', 'collision']):
        events = ea.Scalars(tag)
        if events:
            values = [e.value for e in events]
            print(f"- {tag}: start={values[0]:.4f}, end={values[-1]:.4f}, max={max(values):.4f}, min={min(values):.4f}, len={len(values)}")
