import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
NUM_PARTICLES = 500
WORLD_SIZE = (100, 100)
MOVE_NOISE = 1.0
SENSE_NOISE = 5.0

true_pos = np.array([30.0, 50.0])
particles = np.random.rand(NUM_PARTICLES, 2) * WORLD_SIZE

def move_particles(particles, dx, dy):
    noise = np.random.randn(NUM_PARTICLES, 2) * MOVE_NOISE
    particles += np.array([dx, dy]) + noise
    particles[:, 0] = np.clip(particles[:, 0], 0, WORLD_SIZE[0])
    particles[:, 1] = np.clip(particles[:, 1], 0, WORLD_SIZE[1])
    return particles

def sense(true_pos):
    return true_pos + np.random.randn(2) * SENSE_NOISE

def compute_weights(particles, sensed_pos):
    dists = np.linalg.norm(particles - sensed_pos, axis=1)
    weights = np.exp(-dists**2 / (2 * SENSE_NOISE**2))
    weights += 1.e-300
    weights /= np.sum(weights)
    return weights

def resample(particles, weights):
    indices = np.random.choice(np.arange(NUM_PARTICLES), size=NUM_PARTICLES, p=weights)
    resampled = particles[indices]
    return resampled, indices

# Setup visualization
fig, ax = plt.subplots()
orig_scat = ax.scatter([], [], c='gray', s=5, label='Original Particles')
resampled_scat = ax.scatter([], [], c='blue', s=10, label='Resampled Particles')
robot_dot, = ax.plot([], [], 'ro', label='True Position')
estimated_dot, = ax.plot([], [], 'go', label='Estimated Position')
lines = [ax.plot([], [], 'k-', alpha=0.2)[0] for _ in range(100)]  # limited to 100 lines for performance

def update(frame):
    global particles, true_pos

    dx, dy = 1.0, 0.5
    true_pos += np.array([dx, dy])
    sensed_pos = sense(true_pos)

    particles = move_particles(particles, dx, dy)
    weights = compute_weights(particles, sensed_pos)
    resampled, indices = resample(particles, weights)

    # Update scatter plots
    orig_scat.set_offsets(particles)
    resampled_scat.set_offsets(resampled)
    robot_dot.set_data(true_pos[0], true_pos[1])
    estimated_pos = np.average(resampled, axis=0)
    estimated_dot.set_data(estimated_pos[0], estimated_pos[1])

    # Draw lines from original to resampled (limited to 100 for clarity)
    for i, line in enumerate(lines):
        if i < len(resampled):
            start = particles[indices[i]]
            end = resampled[i]
            line.set_data([start[0], end[0]], [start[1], end[1]])
        else:
            line.set_data([], [])

    particles[:] = resampled  # update particles for next step
    return [orig_scat, resampled_scat, robot_dot, estimated_dot] + lines

ax.set_xlim(0, WORLD_SIZE[0])
ax.set_ylim(0, WORLD_SIZE[1])
ax.set_title("Monte Carlo Localization with Resampling Visualization")
ax.legend()

ani = animation.FuncAnimation(fig, update, frames=50, interval=300, blit=True)
plt.show()
