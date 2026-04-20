"""
Plot trajectory comparison: original vs smart target (tilt50).
Uses dynamics JSON data files.
"""
import json, math, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

def load_traj(path):
    with open(path) as f:
        h = json.load(f)
    steps = h['steps']
    n_agents = len(steps[0]['drone_pos'])
    drone_xy = [np.array([s['drone_pos'][i][:2] for s in steps]) for i in range(n_agents)]
    target_xy = np.array([s['target_pos'][:2] for s in steps])
    drone_z = [np.array([s['drone_pos'][i][2] for s in steps]) for i in range(n_agents)]
    target_z = np.array([s['target_pos'][2] for s in steps])
    target_speed = np.array([s['target_speed'] for s in steps])
    min_dist = np.array([s['min_dist'] for s in steps])
    return {
        'meta': h,
        'drone_xy': drone_xy,
        'target_xy': target_xy,
        'drone_z': drone_z,
        'target_z': target_z,
        'target_speed': target_speed,
        'min_dist': min_dist,
        'n_agents': n_agents,
        'n_steps': len(steps),
    }

def plot_trajectory_panel(ax, data, title, arena_size=3.5):
    """Plot XY trajectory with time-colored lines"""
    n = data['n_steps']
    t_norm = np.linspace(0, 1, n)
    
    # Arena circle
    circle = plt.Circle((0, 0), arena_size, fill=False, color='gray', 
                         linewidth=2, linestyle='--', alpha=0.7)
    ax.add_patch(circle)
    
    # Goal region (approximate)
    goal = plt.Circle((2.0, 0.0), 0.5, fill=True, color='green', alpha=0.15, label='Goal zone')
    ax.add_patch(goal)
    
    # Drone colors
    drone_colors = ['#2196F3', '#FF9800', '#9C27B0']  # blue, orange, purple
    drone_labels = ['Drone 0', 'Drone 1', 'Drone 2']
    
    for i in range(data['n_agents']):
        xy = data['drone_xy'][i]
        # Color gradient from light to dark
        points = xy.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        base_color = matplotlib.colors.to_rgba(drone_colors[i])
        colors = [(*base_color[:3], 0.2 + 0.8 * t) for t in t_norm[:-1]]
        
        lc = LineCollection(segments, colors=colors, linewidths=1.5)
        ax.add_collection(lc)
        
        # Start/end markers
        ax.plot(*xy[0], 'o', color=drone_colors[i], markersize=8, zorder=5)
        ax.plot(*xy[-1], 's', color=drone_colors[i], markersize=10, zorder=5,
                markeredgecolor='black', markeredgewidth=1.5)
        ax.annotate(drone_labels[i], xy[0], fontsize=7, color=drone_colors[i],
                   fontweight='bold', ha='center', va='bottom')
    
    # Target trajectory (red, thicker)
    txy = data['target_xy']
    points = txy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [(1.0, 0.0, 0.0, 0.2 + 0.8 * t) for t in t_norm[:-1]]
    lc = LineCollection(segments, colors=colors, linewidths=3)
    ax.add_collection(lc)
    
    ax.plot(*txy[0], '*', color='red', markersize=15, zorder=6, label='Target start')
    ax.plot(*txy[-1], 'X', color='darkred', markersize=12, zorder=6, 
            markeredgecolor='black', markeredgewidth=1.5, label='Target end')
    
    # Catch radius at endpoint
    if data['meta'].get('done_reason') == 'SUCCESS':
        catch = plt.Circle(txy[-1], 0.4, fill=False, color='red', 
                          linewidth=1.5, linestyle=':', alpha=0.8)
        ax.add_patch(catch)
    
    ax.set_xlim(-arena_size-0.5, arena_size+0.5)
    ax.set_ylim(-arena_size-0.5, arena_size+0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title, fontsize=11, fontweight='bold')

def plot_speed_panel(ax, data, title):
    """Plot speed and distance over time"""
    dt = data['meta']['dt']
    t = np.arange(data['n_steps']) * dt
    
    # Target speed
    ax.plot(t, data['target_speed'], 'r-', linewidth=2, label='Target speed')
    
    # Drone speeds
    drone_colors = ['#2196F3', '#FF9800', '#9C27B0']
    for i in range(data['n_agents']):
        speeds = [data['meta']['steps'][s]['drone_speeds'][i] for s in range(data['n_steps'])]
        ax.plot(t, speeds, color=drone_colors[i], linewidth=1, alpha=0.7, 
                label=f'Drone {i}')
    
    ax.axhline(y=1.5, color='blue', linestyle='--', alpha=0.5, label='Pursuer limit (1.5)')
    ax.axhline(y=2.25, color='red', linestyle='--', alpha=0.5, label='Target cmd limit (2.25)')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

def plot_dist_panel(ax, data, title):
    """Plot min distance over time"""
    dt = data['meta']['dt']
    t = np.arange(data['n_steps']) * dt
    
    ax.plot(t, data['min_dist'], 'k-', linewidth=2)
    ax.axhline(y=0.4, color='red', linestyle='--', label='Catch radius (0.4m)')
    ax.fill_between(t, 0, 0.4, alpha=0.1, color='red')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Min distance (m)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Main
out_dir = sys.argv[1] if len(sys.argv) > 1 else 'analysis/trajectory_plots'
os.makedirs(out_dir, exist_ok=True)

# Load data
files = {}
if os.path.exists('analysis/dynamics/dynamics_seed100.json'):
    files['original'] = load_traj('analysis/dynamics/dynamics_seed100.json')
if os.path.exists('analysis/dynamics/dynamics_seed42.json'):
    files['tilt50'] = load_traj('analysis/dynamics/dynamics_seed42.json')
if os.path.exists('analysis/dynamics/dynamics_smart_seed42.json'):
    files['smart'] = load_traj('analysis/dynamics/dynamics_smart_seed42.json')

n_configs = len(files)
if n_configs == 0:
    print("No dynamics data found!")
    sys.exit(1)

print(f"Plotting {n_configs} configs: {list(files.keys())}")

# === Figure 1: XY Trajectory comparison ===
fig, axes = plt.subplots(1, n_configs, figsize=(7*n_configs, 7))
if n_configs == 1:
    axes = [axes]

titles = {
    'original': f'Original (rep=1.0, tilt=25°)\nseed=100, {files.get("original",{}).get("n_steps",0)} steps, SUCCESS',
    'tilt50': f'Tilt50 (rep=1.5, tilt=50°)\nseed=42, {files.get("tilt50",{}).get("n_steps",0)} steps, SUCCESS',
    'smart': f'Smart (rep=1.5, accel=3.5, tilt=40°)\nseed=42, {files.get("smart",{}).get("n_steps",0)} steps',
}

for i, (key, data) in enumerate(files.items()):
    plot_trajectory_panel(axes[i], data, titles.get(key, key))

fig.suptitle('Pursuit-Evasion Trajectory Comparison (XY Top-Down)', 
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
path1 = os.path.join(out_dir, 'trajectory_comparison_xy.png')
fig.savefig(path1, dpi=200, bbox_inches='tight')
print(f'Saved: {path1}')
plt.close()

# === Figure 2: Speed + distance panels ===
fig, axes = plt.subplots(2, n_configs, figsize=(7*n_configs, 8))
if n_configs == 1:
    axes = axes.reshape(-1, 1)

for i, (key, data) in enumerate(files.items()):
    plot_speed_panel(axes[0, i], data, f'{key}: Speed')
    plot_dist_panel(axes[1, i], data, f'{key}: Min Distance')

fig.suptitle('Speed & Distance Analysis', fontsize=14, fontweight='bold')
fig.tight_layout()
path2 = os.path.join(out_dir, 'speed_distance_comparison.png')
fig.savefig(path2, dpi=200, bbox_inches='tight')
print(f'Saved: {path2}')
plt.close()

# === Figure 3: XZ side view ===
fig, axes = plt.subplots(1, n_configs, figsize=(7*n_configs, 4))
if n_configs == 1:
    axes = [axes]

for i, (key, data) in enumerate(files.items()):
    n = data['n_steps']
    dt = data['meta']['dt']
    t = np.arange(n) * dt
    
    drone_colors = ['#2196F3', '#FF9800', '#9C27B0']
    for j in range(data['n_agents']):
        axes[i].plot(t, data['drone_z'][j], color=drone_colors[j], alpha=0.7, linewidth=1)
    axes[i].plot(t, data['target_z'], 'r-', linewidth=2, label='Target')
    axes[i].set_xlabel('Time (s)')
    axes[i].set_ylabel('Altitude Z (m)')
    axes[i].set_title(f'{key}: Altitude')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

fig.tight_layout()
path3 = os.path.join(out_dir, 'altitude_comparison.png')
fig.savefig(path3, dpi=200, bbox_inches='tight')
print(f'Saved: {path3}')
plt.close()

print("\nAll plots saved!")
