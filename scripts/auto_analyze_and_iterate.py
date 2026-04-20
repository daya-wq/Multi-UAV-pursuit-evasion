#!/usr/bin/env python3
"""
Auto-analyze RL training logs and iterate on hyperparameters.

This script:
1. Parses the training log to extract key metrics
2. Diagnoses network health (KL, grad norm, policy loss, NaN/Inf)
3. Evaluates performance (success rate, out_of_arena, goal_reached, etc.)
4. If performance is poor despite healthy network, adjusts reward/LR params
5. Generates a detailed report and optionally launches a new training run
"""

import argparse
import os
import re
import json
import subprocess
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path


# ============================================================
# 1. Log Parsing
# ============================================================

def parse_log(log_path: str) -> dict:
    """Parse an RL training log file and extract all metrics over time."""
    metrics = defaultdict(list)
    current_iter = 0
    current_frames = 0

    iter_pattern = re.compile(r'(\d+)it \[.*?frames=([0-9.e+]+)')
    metric_pattern = re.compile(r'^([\w/.]+):\s+(.+)$')

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Parse iteration and frames
            m = iter_pattern.search(line)
            if m:
                current_iter = int(m.group(1))
                frames_str = m.group(2)
                current_frames = float(frames_str)

            # Parse metric lines
            m = metric_pattern.match(line)
            if m:
                key = m.group(1)
                val_str = m.group(2)
                try:
                    if val_str == '.nan':
                        val = float('nan')
                    elif val_str == '.inf':
                        val = float('inf')
                    else:
                        val = float(val_str)
                    metrics[key].append({
                        'iter': current_iter,
                        'frames': current_frames,
                        'value': val,
                    })
                except ValueError:
                    pass

    return dict(metrics)


def get_latest_values(metrics: dict, keys: list, n: int = 5) -> dict:
    """Get the average of the last n values for each key."""
    result = {}
    for key in keys:
        if key in metrics and len(metrics[key]) > 0:
            vals = [m['value'] for m in metrics[key][-n:]]
            valid = [v for v in vals if v == v and v != float('inf')]  # exclude nan/inf
            result[key] = sum(valid) / len(valid) if valid else float('nan')
        else:
            result[key] = float('nan')
    return result


def get_trend(metrics: dict, key: str, window: int = 10) -> str:
    """Determine if a metric is rising, falling, or flat over the last window entries."""
    if key not in metrics or len(metrics[key]) < window:
        return 'unknown'
    vals = [m['value'] for m in metrics[key][-window:]]
    valid = [v for v in vals if v == v and v != float('inf')]
    if len(valid) < 3:
        return 'unstable'

    first_half = sum(valid[:len(valid)//2]) / (len(valid)//2)
    second_half = sum(valid[len(valid)//2:]) / (len(valid) - len(valid)//2)

    if second_half > first_half * 1.15:
        return 'rising'
    elif second_half < first_half * 0.85:
        return 'falling'
    else:
        return 'flat'


# ============================================================
# 2. Network Health Diagnosis
# ============================================================

def diagnose_network_health(metrics: dict) -> dict:
    """Check if the network training is numerically healthy."""
    diagnosis = {
        'healthy': True,
        'issues': [],
        'warnings': [],
        'details': {},
    }

    latest = get_latest_values(metrics, [
        'drone/approx_kl',
        'drone/actor_grad_norm',
        'drone/actor_update_skipped',
        'drone/policy_loss',
        'drone/value_loss',
        'drone/entropy',
        'drone/expert_action_mse',
        'drone/expert_action_mse_ema',
        'drone/actor_log_std_mean',
        'drone/ESS',
        'drone/TP_loss',
        'drone/explained_var',
    ], n=10)

    diagnosis['details'] = latest

    # Check for NaN/Inf
    for key, val in latest.items():
        if val != val or val == float('inf'):
            diagnosis['healthy'] = False
            diagnosis['issues'].append(f'{key} is NaN/Inf')

    # Check approx_kl
    kl = latest.get('drone/approx_kl', 0)
    if kl == kl:
        if kl > 0.1:
            diagnosis['healthy'] = False
            diagnosis['issues'].append(f'approx_kl={kl:.4f} >> 0.03 threshold, policy updates unstable')
        elif kl > 0.03:
            diagnosis['warnings'].append(f'approx_kl={kl:.4f} near threshold')

    # Check actor_update_skipped
    skip = latest.get('drone/actor_update_skipped', 0)
    if skip == skip and skip > 0.5:
        diagnosis['healthy'] = False
        diagnosis['issues'].append(f'actor_update_skipped={skip:.2%}, most updates are being skipped')

    # Check grad norm
    grad = latest.get('drone/actor_grad_norm', 0)
    if grad == grad and grad != float('inf'):
        diagnosis['details']['grad_norm_clipped'] = grad > 10.0
        if grad > 100:
            diagnosis['warnings'].append(f'actor_grad_norm={grad:.1f}, extreme gradient clipping')
        elif grad > 10:
            diagnosis['warnings'].append(f'actor_grad_norm={grad:.1f}, persistent gradient clipping (max_grad_norm=10)')

    # Check ESS
    ess = latest.get('drone/ESS', 1.0)
    if ess == ess and ess < 0.5:
        diagnosis['warnings'].append(f'ESS={ess:.3f}, low effective sample size')

    # Check explained_var
    ev = latest.get('drone/explained_var', 1.0)
    if ev == ev and ev < 0.3:
        diagnosis['warnings'].append(f'explained_var={ev:.3f}, critic poorly predicting returns')

    # Check expert MSE trend
    mse_trend = get_trend(metrics, 'drone/expert_action_mse', window=10)
    if mse_trend == 'rising':
        diagnosis['warnings'].append(f'expert_action_mse is rising, policy diverging from expert')

    return diagnosis


# ============================================================
# 3. Performance Diagnosis
# ============================================================

def diagnose_performance(metrics: dict) -> dict:
    """Analyze task-level performance metrics."""
    diagnosis = {
        'acceptable': False,
        'failure_mode': 'unknown',
        'issues': [],
        'recommendations': [],
        'details': {},
    }

    # Get latest eval stats (prefer eval, fall back to train)
    eval_keys = [
        'eval/stats.success', 'eval/stats.out_of_arena', 'eval/stats.goal_reached',
        'eval/stats.collision_drone', 'eval/stats.first_capture_step',
        'eval/stats.role_reward', 'eval/stats.close_reward',
        'eval/stats.phi_block', 'eval/stats.phi_pressure', 'eval/stats.phi_spread',
        'eval/stats.n_agents_ahead_mean',
    ]
    train_keys = [k.replace('eval/', 'train/') for k in eval_keys]

    eval_latest = get_latest_values(metrics, eval_keys, n=3)
    train_latest = get_latest_values(metrics, train_keys, n=5)

    # Use eval if available, else train
    success = eval_latest.get('eval/stats.success', float('nan'))
    if success != success:
        success = train_latest.get('train/stats.success', 0)
        using = 'train'
    else:
        using = 'eval'

    out_of_arena = eval_latest.get('eval/stats.out_of_arena',
                   train_latest.get('train/stats.out_of_arena', float('nan')))
    goal_reached = eval_latest.get('eval/stats.goal_reached',
                   train_latest.get('train/stats.goal_reached', float('nan')))
    collision_drone = eval_latest.get('eval/stats.collision_drone',
                     train_latest.get('train/stats.collision_drone', float('nan')))
    phi_block = eval_latest.get('eval/stats.phi_block',
                train_latest.get('train/stats.phi_block', float('nan')))
    n_ahead = eval_latest.get('eval/stats.n_agents_ahead_mean',
              train_latest.get('train/stats.n_agents_ahead_mean', float('nan')))
    role_reward = eval_latest.get('eval/stats.role_reward',
                  train_latest.get('train/stats.role_reward', float('nan')))

    diagnosis['details'] = {
        'success': success,
        'out_of_arena': out_of_arena,
        'goal_reached': goal_reached,
        'collision_drone': collision_drone,
        'phi_block': phi_block,
        'n_agents_ahead_mean': n_ahead,
        'role_reward': role_reward,
        'metric_source': using,
    }

    # Performance thresholds
    if success == success and success > 0.30:
        diagnosis['acceptable'] = True
        return diagnosis

    # Identify failure modes
    if out_of_arena == out_of_arena and out_of_arena > 0.5:
        diagnosis['failure_mode'] = 'out_of_arena'
        diagnosis['issues'].append(
            f'out_of_arena={out_of_arena:.1%} — drones frequently fly out of arena'
        )
        diagnosis['recommendations'].append(
            'Increase collision_coef to penalize wall collisions more heavily'
        )
        diagnosis['recommendations'].append(
            'Consider reducing actor_lr to stabilize flight behavior'
        )

    if goal_reached == goal_reached and goal_reached > 0.3 and (success == success and success < 0.1):
        diagnosis['failure_mode'] = 'interception_failure'
        diagnosis['issues'].append(
            f'goal_reached={goal_reached:.1%} but success={success:.1%} — target reaching goal, pursuers not intercepting'
        )
        diagnosis['recommendations'].append(
            'Increase role_reward_coef to encourage waypoint following'
        )
        diagnosis['recommendations'].append(
            'Increase goal_penalty_coef to make goal defense more urgent'
        )

    if phi_block == phi_block and phi_block < 0.05:
        diagnosis['issues'].append(
            f'phi_block={phi_block:.3f} — no effective blocking formation'
        )
        diagnosis['recommendations'].append(
            'Enable capture_progress_coef to reward closing distance'
        )

    if role_reward == role_reward and role_reward < -0.05:
        diagnosis['issues'].append(
            f'role_reward={role_reward:.4f} — agents moving away from role waypoints'
        )

    # Check success trend
    train_success_trend = get_trend(metrics, 'train/stats.success', window=10)
    out_of_arena_trend = get_trend(metrics, 'train/stats.out_of_arena', window=10)

    diagnosis['details']['success_trend'] = train_success_trend
    diagnosis['details']['out_of_arena_trend'] = out_of_arena_trend

    if train_success_trend == 'rising':
        diagnosis['issues'].append('Success rate is trending upward — training may just need more time')
    elif train_success_trend == 'flat' and success < 0.1:
        diagnosis['issues'].append('Success rate is flat at a very low level — intervention needed')

    return diagnosis


# ============================================================
# 4. Parameter Adjustment Logic
# ============================================================

def compute_adjustments(net_diag: dict, perf_diag: dict, current_params: dict) -> dict:
    """Based on diagnoses, compute parameter adjustments for the next run."""
    adjustments = {}
    reasons = []

    if not net_diag['healthy']:
        # Network issues — be conservative
        if any('approx_kl' in issue for issue in net_diag['issues']):
            adjustments['algo.actor.lr'] = max(current_params.get('actor_lr', 5e-5) / 3, 2e-6)
            reasons.append(f"KL too high, reducing actor_lr to {adjustments['algo.actor.lr']:.1e}")

        if any('skipped' in issue for issue in net_diag['issues']):
            adjustments['algo.actor.lr'] = max(current_params.get('actor_lr', 5e-5) / 5, 2e-6)
            reasons.append(f"Updates being skipped, reducing actor_lr to {adjustments['algo.actor.lr']:.1e}")

        return {'adjustments': adjustments, 'reasons': reasons, 'strategy': 'fix_network'}

    # Network healthy, fix performance
    failure_mode = perf_diag.get('failure_mode', 'unknown')

    if failure_mode == 'out_of_arena':
        # Drones flying out — need stronger boundary awareness
        current_collision = current_params.get('collision_coef', 40.0)
        adjustments['task.collision_coef'] = min(current_collision * 1.5, 80.0)
        reasons.append(f"out_of_arena high, raising collision_coef: {current_collision} → {adjustments['task.collision_coef']}")

        # Also reduce LR slightly to stabilize flight
        current_lr = current_params.get('actor_lr', 5e-5)
        adjustments['algo.actor.lr'] = max(current_lr * 0.5, 5e-6)
        reasons.append(f"Reducing actor_lr for stability: {current_lr:.1e} → {adjustments['algo.actor.lr']:.1e}")

        # Pull role_reward back to not overwhelm safe flight learning
        adjustments['task.expert2_role_reward_coef'] = 0.3
        adjustments['task.expert2_close_reward_coef'] = 0.3
        reasons.append("Reducing role/close reward to 0.3 to prioritize safe flight")

    elif failure_mode == 'interception_failure':
        # Target reaching goal — need better interception
        current_goal_penalty = current_params.get('goal_penalty_coef', 40.0)
        adjustments['task.goal_penalty_coef'] = min(current_goal_penalty * 1.25, 60.0)
        reasons.append(f"Goal defense failing, raising goal_penalty_coef: {current_goal_penalty} → {adjustments['task.goal_penalty_coef']}")

        # Boost role reward to encourage waypoint following
        adjustments['task.expert2_role_reward_coef'] = 0.5
        adjustments['task.expert2_close_reward_coef'] = 0.5
        reasons.append("Increasing role/close reward to 0.5 for stronger interception")

        # Enable capture progress for dense pursuit signal
        adjustments['task.capture_progress_coef'] = 0.1
        reasons.append("Enabling capture_progress_coef=0.1 for dense chase reward")

    else:
        # General poor performance
        current_lr = current_params.get('actor_lr', 5e-5)
        adjustments['algo.actor.lr'] = max(current_lr * 0.7, 5e-6)
        reasons.append(f"General poor performance, moderately reducing actor_lr: {current_lr:.1e} → {adjustments['algo.actor.lr']:.1e}")

    return {'adjustments': adjustments, 'reasons': reasons, 'strategy': failure_mode}


# ============================================================
# 5. Report Generation
# ============================================================

def generate_report(
    log_path: str,
    metrics: dict,
    net_diag: dict,
    perf_diag: dict,
    adj: dict,
    report_dir: str,
) -> str:
    """Generate a detailed markdown report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(report_dir, f'auto_analysis_{timestamp}.md')

    total_iters = 0
    total_frames = 0
    if 'drone/approx_kl' in metrics and metrics['drone/approx_kl']:
        total_iters = metrics['drone/approx_kl'][-1]['iter']
        total_frames = metrics['drone/approx_kl'][-1]['frames']

    lines = []
    lines.append(f'# 自动训练分析报告')
    lines.append(f'')
    lines.append(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**分析日志**: `{log_path}`')
    lines.append(f'**总迭代数**: {total_iters}')
    lines.append(f'**总帧数**: {total_frames:.2e}')
    lines.append(f'')

    # Network Health
    lines.append(f'## 1. 网络健康状态')
    lines.append(f'')
    health_icon = '✅' if net_diag['healthy'] else '❌'
    lines.append(f'**状态**: {health_icon} {"健康" if net_diag["healthy"] else "异常"}')
    lines.append(f'')

    lines.append(f'| 指标 | 值 | 状态 |')
    lines.append(f'|:---|:---|:---|')
    for key, val in net_diag['details'].items():
        if isinstance(val, bool):
            status = '⚠️' if val else '✅'
            lines.append(f'| `{key}` | {val} | {status} |')
        elif isinstance(val, float):
            if val != val:
                lines.append(f'| `{key}` | NaN | ❌ |')
            elif val == float('inf'):
                lines.append(f'| `{key}` | Inf | ❌ |')
            else:
                lines.append(f'| `{key}` | {val:.6f} | |')
    lines.append(f'')

    if net_diag['issues']:
        lines.append(f'### 问题')
        for issue in net_diag['issues']:
            lines.append(f'- ❌ {issue}')
        lines.append(f'')

    if net_diag['warnings']:
        lines.append(f'### 警告')
        for warn in net_diag['warnings']:
            lines.append(f'- ⚠️ {warn}')
        lines.append(f'')

    # Performance
    lines.append(f'## 2. 性能评估')
    lines.append(f'')
    perf_icon = '✅' if perf_diag['acceptable'] else '❌'
    lines.append(f'**状态**: {perf_icon} {"达标 (success > 30%)" if perf_diag["acceptable"] else "未达标"}')
    lines.append(f'**主要失败模式**: `{perf_diag["failure_mode"]}`')
    lines.append(f'')

    lines.append(f'| 指标 | 值 |')
    lines.append(f'|:---|:---|')
    for key, val in perf_diag['details'].items():
        if isinstance(val, float) and val == val:
            lines.append(f'| `{key}` | {val:.4f} |')
        else:
            lines.append(f'| `{key}` | {val} |')
    lines.append(f'')

    if perf_diag['issues']:
        lines.append(f'### 问题分析')
        for issue in perf_diag['issues']:
            lines.append(f'- {issue}')
        lines.append(f'')

    if perf_diag['recommendations']:
        lines.append(f'### 建议')
        for rec in perf_diag['recommendations']:
            lines.append(f'- 💡 {rec}')
        lines.append(f'')

    # Success rate history
    if 'train/stats.success' in metrics:
        lines.append(f'### 训练成功率历史')
        lines.append(f'```')
        for entry in metrics['train/stats.success'][-20:]:
            lines.append(f"  iter {entry['iter']:>5d} | frames {entry['frames']:.2e} | success = {entry['value']:.4f}")
        lines.append(f'```')
        lines.append(f'')

    if 'eval/stats.success' in metrics:
        lines.append(f'### 评估成功率历史')
        lines.append(f'```')
        for entry in metrics['eval/stats.success']:
            lines.append(f"  iter {entry['iter']:>5d} | frames {entry['frames']:.2e} | success = {entry['value']:.4f}")
        lines.append(f'```')
        lines.append(f'')

    # Adjustments
    lines.append(f'## 3. 自动调参决策')
    lines.append(f'')
    lines.append(f'**策略**: `{adj["strategy"]}`')
    lines.append(f'')

    if adj['adjustments']:
        lines.append(f'| 参数 | 新值 |')
        lines.append(f'|:---|:---|')
        for key, val in adj['adjustments'].items():
            lines.append(f'| `{key}` | {val} |')
        lines.append(f'')

        lines.append(f'### 调参理由')
        for reason in adj['reasons']:
            lines.append(f'- {reason}')
    else:
        lines.append(f'无需调参。')
    lines.append(f'')

    # Write report
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    return report_path


# ============================================================
# 6. Training Script Generation
# ============================================================

def find_best_checkpoint(checkpoint_dir: str, log_metrics: dict) -> str:
    """Find the best checkpoint based on eval success rate."""
    # Get all checkpoints sorted by frame count
    if not os.path.isdir(checkpoint_dir):
        return None

    # Find the run directory (most recent HideAndSeek_*)
    run_dirs = sorted([
        d for d in os.listdir(checkpoint_dir)
        if d.startswith('HideAndSeek_')
    ])

    if not run_dirs:
        return None

    latest_dir = os.path.join(checkpoint_dir, run_dirs[-1])
    checkpoints = sorted([
        f for f in os.listdir(latest_dir)
        if f.endswith('.pt')
    ])

    if not checkpoints:
        return None

    # Return the latest checkpoint (could be improved with eval success mapping)
    return os.path.join(latest_dir, checkpoints[-1])


def generate_training_script(
    script_dir: str,
    checkpoint_path: str,
    adjustments: dict,
    version: str = 'v8',
    base_params: dict = None,
) -> str:
    """Generate a new training script with adjusted parameters."""
    if base_params is None:
        base_params = {}

    script_path = os.path.join(script_dir, f'run_{version}.sh')

    # Build parameter overrides
    params = {
        'task': 'HideAndSeek',
        'headless': 'true',
        'wandb.mode': 'disabled',
        'model_dir': checkpoint_path,
        'algo.use_TP_net': '1',
        'algo.warmstart.actor_freeze.enabled': 'false',
        'algo.warmstart.actor_lr_warmup.enabled': 'false',
        'algo.actor.lr': str(base_params.get('actor_lr', 5e-5)),
        'algo.actor.log_std_init': '-2.0',
        'algo.actor.log_std_min': '-2.0',
        'algo.actor.log_std_max': '-1.0',
        'algo.warmstart.actor_log_std_schedule.enabled': 'false',
        'algo.warmstart.expert_kl.enabled': 'true',
        'algo.warmstart.expert_kl.coef_init': '0.3',
        'algo.warmstart.expert_kl.coef_final': '0.05',
        'algo.warmstart.expert_kl.decay_frames': '300000000',
        'algo.warmstart.expert_kl.mse_boost_enabled': 'false',
        'algo.entropy_coef': '0.001',
        'algo.clip_param': '0.1',
        'algo.train_every': str(base_params.get('train_every', 128)),
        'task.env.num_envs': '3072',
        'task.v_prey': '1.5',
        'task.v_drone': '1.5',
        'task.expert2_role_reward_coef': str(base_params.get('role_reward_coef', 0.3)),
        'task.expert2_close_reward_coef': str(base_params.get('close_reward_coef', 0.3)),
        'total_frames': '500000000',
        'max_iters': '20000',
        'eval_interval': '50',
        'save_interval': '200',
        'seed': '42',
    }

    # Apply adjustments (override base params)
    for key, val in adjustments.items():
        params[key] = str(val)

    log_file = f'analysis/rl_warmstart_{version}.log'

    # Build the override string for documentation
    adj_doc = '\n'.join([f'#   {k}: {v}' for k, v in adjustments.items()])

    script_content = f"""#!/bin/bash
# RL warmstart {version}: auto-generated by auto_analyze_and_iterate.py
# Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#
# Adjustments from previous run:
{adj_doc}
#
# Checkpoint: {checkpoint_path}

set -euo pipefail
cd "$(dirname "$0")/.."

LOG_FILE="{log_file}"

nohup bash -c '
source /home/uavlab/miniconda3/etc/profile.d/conda.sh
conda activate sim
source setup_conda_env.sh

python scripts/train.py \\
"""

    for key, val in params.items():
        script_content += f'  {key}={val} \\\n'

    # Remove trailing backslash from last line
    script_content = script_content.rstrip(' \\\n') + '\n'

    script_content += f"""' > "$LOG_FILE" 2>&1 &

echo "{version} launched! PID=$!"
echo "Log: $LOG_FILE"
echo "Tail: tail -f $LOG_FILE"
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)
    return script_path


# ============================================================
# 7. Clean and Restart
# ============================================================

def clean_training_env():
    """Kill any stale training processes."""
    cmds = [
        'pkill -9 -f "scripts/train.py" || true',
        'pkill -9 -u uavlab -f kit || true',
    ]
    for cmd in cmds:
        subprocess.run(cmd, shell=True, capture_output=True)

    # Wait for processes to die
    import time
    time.sleep(5)

    # Check GPU
    result = subprocess.run('nvidia-smi --query-compute-apps=pid,name,used_gpu_memory --format=csv',
                          shell=True, capture_output=True, text=True)
    print(f"[GPU Status]\n{result.stdout}")


def launch_training(script_path: str):
    """Launch a new training run."""
    print(f"[{datetime.now()}] Launching: bash {script_path}")
    result = subprocess.run(
        f'bash {script_path}',
        shell=True, capture_output=True, text=True,
        cwd=os.path.dirname(os.path.dirname(script_path)),
    )
    print(f"stdout: {result.stdout}")
    if result.returncode != 0:
        print(f"stderr: {result.stderr}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Auto-analyze RL training and iterate')
    parser.add_argument('--log-file', required=True, help='Path to training log file')
    parser.add_argument('--report-dir', default='analysis/auto_monitor_reports')
    parser.add_argument('--checkpoint-dir', default='checkpoints')
    parser.add_argument('--scripts-dir', default='scripts')
    parser.add_argument('--config-file', default='cfg/task/HideAndSeek.yaml')
    parser.add_argument('--auto-restart', action='store_true',
                       help='Automatically restart training with new params if needed')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only generate report, do not restart training')
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)

    print(f"[{datetime.now()}] Parsing log: {args.log_file}")
    metrics = parse_log(args.log_file)

    if not metrics:
        print("ERROR: No metrics found in log file!")
        sys.exit(1)

    # Current params (corrected v7 — identical to v5 except actor_lr)
    current_params = {
        'actor_lr': 1.5e-5,
        'train_every': 64,
        'role_reward_coef': 0.3,
        'close_reward_coef': 0.3,
        'collision_coef': 40.0,
        'goal_penalty_coef': 40.0,
    }

    print(f"[{datetime.now()}] Diagnosing network health...")
    net_diag = diagnose_network_health(metrics)
    print(f"  Network healthy: {net_diag['healthy']}")
    if net_diag['issues']:
        for issue in net_diag['issues']:
            print(f"  ❌ {issue}")
    if net_diag['warnings']:
        for warn in net_diag['warnings']:
            print(f"  ⚠️ {warn}")

    print(f"\n[{datetime.now()}] Diagnosing performance...")
    perf_diag = diagnose_performance(metrics)
    print(f"  Performance acceptable: {perf_diag['acceptable']}")
    print(f"  Failure mode: {perf_diag['failure_mode']}")
    for issue in perf_diag['issues']:
        print(f"  - {issue}")

    print(f"\n[{datetime.now()}] Computing adjustments...")
    adj = compute_adjustments(net_diag, perf_diag, current_params)
    print(f"  Strategy: {adj['strategy']}")
    for reason in adj['reasons']:
        print(f"  - {reason}")

    print(f"\n[{datetime.now()}] Generating report...")
    report_path = generate_report(
        args.log_file, metrics, net_diag, perf_diag, adj, args.report_dir
    )
    print(f"  Report saved to: {report_path}")

    if perf_diag['acceptable']:
        print(f"\n[{datetime.now()}] Performance is acceptable! No restart needed.")
        return

    if args.dry_run:
        print(f"\n[{datetime.now()}] Dry run — not restarting training.")
        return

    if args.auto_restart and adj['adjustments']:
        print(f"\n[{datetime.now()}] Finding best checkpoint...")
        best_ckpt = find_best_checkpoint(args.checkpoint_dir, metrics)
        if best_ckpt is None:
            print("  ERROR: No checkpoint found!")
            return
        print(f"  Using checkpoint: {best_ckpt}")

        print(f"\n[{datetime.now()}] Generating new training script...")
        script_path = generate_training_script(
            args.scripts_dir,
            best_ckpt,
            adj['adjustments'],
            version='v8',
            base_params=current_params,
        )
        print(f"  Script saved to: {script_path}")

        print(f"\n[{datetime.now()}] Cleaning training environment...")
        clean_training_env()

        print(f"\n[{datetime.now()}] Launching new training run...")
        launch_training(script_path)

        print(f"\n[{datetime.now()}] Auto-iteration complete!")
    else:
        print(f"\n[{datetime.now()}] No auto-restart. Review the report and adjust manually.")


if __name__ == '__main__':
    main()
