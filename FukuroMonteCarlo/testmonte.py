import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameter simulasi
NUM_PARTICLES = 500
WORLD_SIZE = (100, 100)

# Sensor dan gerakan noise
MOVE_NOISE = 1.0
SENSE_NOISE = 5.0

# Posisi robot sebenarnya
true_pos = np.array([30.0, 50.0])

# Inisialisasi partikel
particles = np.random.rand(NUM_PARTICLES, 2) * WORLD_SIZE

def move_particles(particles, dx, dy):
    """Gerakkan partikel dengan noise"""
    noise = np.random.randn(NUM_PARTICLES, 2) * MOVE_NOISE
    particles += np.array([dx, dy]) + noise
    particles[:, 0] = np.clip(particles[:, 0], 0, WORLD_SIZE[0])
    particles[:, 1] = np.clip(particles[:, 1], 0, WORLD_SIZE[1])
    return particles

def sense(true_pos):
    """Sensor membaca posisi dengan noise"""
    return true_pos + np.random.randn(2) * SENSE_NOISE

def compute_weights(particles, sensed_pos):
    """Hitung bobot berdasarkan jarak ke sensed position"""
    dists = np.linalg.norm(particles - sensed_pos, axis=1)
    weights = np.exp(-dists**2 / (2 * SENSE_NOISE**2))
    weights += 1.e-300  # Hindari nol
    weights /= np.sum(weights)
    return weights

def resample(particles, weights):
    """Resample partikel berdasarkan bobot"""
    indices = np.random.choice(np.arange(NUM_PARTICLES), size=NUM_PARTICLES, p=weights)
    return particles[indices]

# Setup visualisasi
fig, ax = plt.subplots()
scat = ax.scatter(particles[:, 0], particles[:, 1], c='gray', s=2)
robot_dot, = ax.plot([], [], 'ro', label="True Position")
estimated_dot, = ax.plot([], [], 'bo', label="Estimated")

def update(frame):
    global particles, true_pos

    # Simulasi gerakan robot
    dx, dy = 1.0, 0.5
    true_pos += np.array([dx, dy])

    # Sensor membaca posisi (dengan noise)
    sensed_pos = sense(true_pos)

    # Gerakkan partikel
    particles = move_particles(particles, dx, dy)

    # Hitung bobot dan resample
    weights = compute_weights(particles, sensed_pos)
    particles = resample(particles, weights)

    # Update visual
    scat.set_offsets(particles)
    robot_dot.set_data(true_pos[0], true_pos[1])
    estimated_pos = np.average(particles, axis=0, weights=weights)
    estimated_dot.set_data(estimated_pos[0], estimated_pos[1])
    return scat, robot_dot, estimated_dot

ax.set_xlim(0, WORLD_SIZE[0])
ax.set_ylim(0, WORLD_SIZE[1])
ax.set_title("Monte Carlo Localization")
ax.legend()

ani = animation.FuncAnimation(fig, update, frames=50, interval=200, blit=True)
plt.show()
