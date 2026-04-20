"""
Plot 3D trajectory comparison: original vs tilt50.
"""
import json, math, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

def load_traj(path):
    with open(path) as f:
        h = json.load(f)
    steps = h['steps']
    n_agents = len(steps[0]['drone_pos'])
    drone_xyz = [np.array([s['drone_pos'][i] for s in steps]) for i in range(n_agents)]
    target_xyz = np.array([s['target_pos'] for s in steps])
    target_speed = np.array([s['target_speed'] for s in steps])
    drone_speeds = [np.array([s['drone_speeds'][i] for s in steps]) for i in range(n_agents)]
    min_dist = np.array([s['min_dist'] for s in steps])
    return {
        'meta': h,
        'drone_xyz': drone_xyz,
        'target_xyz': target_xyz,
        'drone_speeds': drone_speeds,
        'target_speed': target_speed,
        'min_dist': min_dist,
        'n_agents': n_agents,
        'n_steps': len(steps),
    }

def draw_arena_wireframe(ax, arena_size=3.5, max_height=5.0, min_z=0.2, n_pts=80):
    """Draw arena cylinder wireframe"""
    theta = np.linspace(0, 2*np.pi, n_pts)
    x = arena_size * np.cos(theta)
    y = arena_size * np.sin(theta)
    # Bottom circle
    ax.plot(x, y, np.full_like(x, min_z), color='gray', alpha=0.3, linewidth=1)
    # Top circle
    ax.plot(x, y, np.full_like(x, max_height), color='gray', alpha=0.3, linewidth=1)
    # Vertical pillars
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        px, py = arena_size * np.cos(angle), arena_size * np.sin(angle)
        ax.plot([px, px], [py, py], [min_z, max_height], color='gray', alpha=0.15, linewidth=0.8)
    # Ground grid
    for r in [1.0, 2.0, 3.0, 3.5]:
        xr = r * np.cos(theta)
        yr = r * np.sin(theta)
        ax.plot(xr, yr, np.full_like(xr, min_z), color='gray', alpha=0.08, linewidth=0.5)

def plot_3d_trajectory(ax, data, title, arena_size=3.5):
    """Plot 3D trajectory with time-gradient coloring"""
    n = data['n_steps']
    
    # Draw arena
    draw_arena_wireframe(ax, arena_size)
    
    # Drone colors
    drone_colors_hex = ['#2196F3', '#FF9800', '#9C27B0']
    drone_labels = ['Pursuer 0', 'Pursuer 1', 'Pursuer 2']
    
    for i in range(data['n_agents']):
        xyz = data['drone_xyz'][i]
        # Line with gradient
        points = xyz.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        base_rgba = matplotlib.colors.to_rgba(drone_colors_hex[i])
        colors = [(*base_rgba[:3], 0.15 + 0.85 * (t / n)) for t in range(n - 1)]
        
        lc = Line3DCollection(segments, colors=colors, linewidths=1.8)
        ax.add_collection3d(lc)
        
        # Start marker (circle)
        ax.scatter(*xyz[0], color=drone_colors_hex[i], s=60, marker='o', zorder=5,
                   edgecolors='white', linewidth=0.8, label=f'{drone_labels[i]} start')
        # End marker (square)
        ax.scatter(*xyz[-1], color=drone_colors_hex[i], s=100, marker='s', zorder=5,
                   edgecolors='black', linewidth=1.2)
    
    # Target trajectory (red, thicker)
    tgt = data['target_xyz']
    points = tgt.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colors = [(1.0, 0.0, 0.0, 0.15 + 0.85 * (t / n)) for t in range(n - 1)]
    lc = Line3DCollection(segments, colors=colors, linewidths=3.0)
    ax.add_collection3d(lc)
    
    # Target start/end
    ax.scatter(*tgt[0], color='red', s=150, marker='*', zorder=6, label='Target start')
    ax.scatter(*tgt[-1], color='darkred', s=120, marker='X', zorder=6,
               edgecolors='black', linewidth=1.2, label='Target end')
    
    # Catch sphere at end (wireframe)
    if data['meta'].get('done_reason') == 'SUCCESS':
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        r = 0.4
        cx, cy, cz = tgt[-1]
        xs = cx + r * np.outer(np.cos(u), np.sin(v))
        ys = cy + r * np.outer(np.sin(u), np.sin(v))
        zs = cz + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(xs, ys, zs, color='red', alpha=0.08)
    
    ax.set_xlabel('X (m)', fontsize=9)
    ax.set_ylabel('Y (m)', fontsize=9)
    ax.set_zlabel('Z (m)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # Set limits
    ax.set_xlim(-arena_size - 0.3, arena_size + 0.3)
    ax.set_ylim(-arena_size - 0.3, arena_size + 0.3)
    ax.set_zlim(0, 5.0)
    
    # Nice viewing angle
    ax.view_init(elev=25, azim=-60)
    ax.tick_params(labelsize=7)

# Main
out_dir = sys.argv[1] if len(sys.argv) > 1 else 'analysis/trajectory_plots'
os.makedirs(out_dir, exist_ok=True)

# Load data
configs = {}
candidates = [
    ('Original\n(rep=1.0, tilt=25°)', 'analysis/dynamics/dynamics_original_seed100.json', 'analysis/dynamics/dynamics_seed100.json'),
    ('Tilt50\n(rep=1.5, tilt=50°)', 'analysis/dynamics/dynamics_tilt50_seed42.json', 'analysis/dynamics/dynamics_seed42.json'),
    ('Smart Target\n(rep=1.5, accel=3.5, tilt=40°)', 'analysis/dynamics/dynamics_smart_seed42.json', None),
]

for label, path1, path2 in candidates:
    p = path1 if os.path.exists(path1) else (path2 if path2 and os.path.exists(path2) else None)
    if p:
        configs[label] = load_traj(p)
        print(f"  Loaded: {label} <- {p} ({configs[label]['n_steps']} steps)")

n = len(configs)
if n == 0:
    print("No data found!"); sys.exit(1)

# === Figure 1: 3D trajectory panels ===
fig = plt.figure(figsize=(8 * n, 8))
for i, (label, data) in enumerate(configs.items()):
    ax = fig.add_subplot(1, n, i + 1, projection='3d')
    seed = data['meta'].get('seed', '?')
    steps = data['n_steps']
    reason = data['meta'].get('done_reason', 'UNKNOWN')
    plot_3d_trajectory(ax, data, f'{label}\nseed={seed}, {steps} steps, {reason}')

fig.suptitle('3D Pursuit-Evasion Trajectory Comparison', fontsize=16, fontweight='bold', y=1.0)
fig.tight_layout()
path1 = os.path.join(out_dir, 'trajectory_3d_comparison.png')
fig.savefig(path1, dpi=200, bbox_inches='tight')
print(f'\nSaved: {path1}')
plt.close()

# === Figure 2: Multiple viewing angles for each config ===
angles = [(25, -60, 'Perspective'), (90, -90, 'Top-Down (XY)'), (0, 0, 'Front (XZ)'), (0, -90, 'Side (YZ)')]

for label, data in configs.items():
    clean_label = label.split('\n')[0].replace(' ', '_').replace('°', 'deg')
    fig = plt.figure(figsize=(24, 6))
    for j, (elev, azim, view_name) in enumerate(angles):
        ax = fig.add_subplot(1, 4, j + 1, projection='3d')
        seed = data['meta'].get('seed', '?')
        plot_3d_trajectory(ax, data, f'{view_name}')
        ax.view_init(elev=elev, azim=azim)
    
    fig.suptitle(f'{label} — seed={data["meta"].get("seed","?")} — Multi-View', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = os.path.join(out_dir, f'trajectory_3d_{clean_label}_multiview.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'Saved: {path}')
    plt.close()

# === Figure 3: Speed + Distance (same as before but together) ===
fig, axes = plt.subplots(2, n, figsize=(7 * n, 8))
if n == 1:
    axes = axes.reshape(-1, 1)

for i, (label, data) in enumerate(configs.items()):
    dt = data['meta']['dt']
    t = np.arange(data['n_steps']) * dt
    
    # Speed panel
    ax = axes[0, i]
    ax.plot(t, data['target_speed'], 'r-', linewidth=2.5, label='Target')
    drone_c = ['#2196F3', '#FF9800', '#9C27B0']
    for j in range(data['n_agents']):
        ax.plot(t, data['drone_speeds'][j], color=drone_c[j], linewidth=1, alpha=0.8, 
                label=f'Drone {j}')
    ax.axhline(1.5, color='blue', ls='--', alpha=0.4, label='Pursuer PhysX limit')
    ax.axhline(2.25, color='red', ls='--', alpha=0.4, label='Target cmd limit')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title(f'{label.split(chr(10))[0]}: Speed')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Distance panel
    ax = axes[1, i]
    ax.plot(t, data['min_dist'], 'k-', linewidth=2)
    ax.axhline(0.4, color='red', ls='--', label='Catch (0.4m)')
    ax.fill_between(t, 0, 0.4, alpha=0.1, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Min dist (m)')
    ax.set_title(f'{label.split(chr(10))[0]}: Distance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Speed & Distance Analysis', fontsize=14, fontweight='bold')
fig.tight_layout()
path3 = os.path.join(out_dir, 'speed_distance_comparison.png')
fig.savefig(path3, dpi=200, bbox_inches='tight')
print(f'Saved: {path3}')
plt.close()

print('\nAll 3D plots saved!')
