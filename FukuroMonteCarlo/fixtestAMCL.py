import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation settings
NUM_PARTICLES = 100
MAP_SIZE = (100, 100)
MOVE_DISTANCE = 5.0
NOISE_STD = 2.0

# Ground truth robot
robot_position = np.array([20.0, 20.0])

# Particle class
class Particle:
    def __init__(self, x, y, weight=1.0):
        self.x = x
        self.y = y
        self.weight = weight

    def move(self, dx, dy):
        self.x += dx + np.random.normal(0, NOISE_STD)
        self.y += dy + np.random.normal(0, NOISE_STD)

# Initialize particles randomly
particles = [Particle(np.random.uniform(0, MAP_SIZE[0]),
                      np.random.uniform(0, MAP_SIZE[1]))
             for _ in range(NUM_PARTICLES)]

# Visualization setup
fig, ax = plt.subplots(figsize=(6, 6))
sc_particles = ax.scatter([], [], c='blue', label='Particles')
sc_robot = ax.scatter([], [], c='red', s=100, label='Robot')
step_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def sense(particle, robot_pos):
    # In this toy model, we just use Euclidean distance as a "likelihood"
    dist = np.linalg.norm(np.array([particle.x, particle.y]) - robot_pos)
    return np.exp(-dist**2 / (2 * NOISE_STD**2))

def resample(particles):
    weights = np.array([p.weight for p in particles])
    weights += 1e-300  # prevent division by zero
    weights /= np.sum(weights)
    indices = np.random.choice(len(particles), size=len(particles), p=weights)
    return [Particle(particles[i].x, particles[i].y) for i in indices]

# Simulation step manager
state = ['predict', 'sense', 'resample', 'move']
step_counter = 0

def update(frame):
    global particles, robot_position, step_counter
    ax.clear()
    ax.set_xlim(0, MAP_SIZE[0])
    ax.set_ylim(0, MAP_SIZE[1])

    current_step = state[step_counter % len(state)]

    if current_step == 'predict':
        # 1. Prediction step: simulate motion
        for p in particles:
            p.move(MOVE_DISTANCE, 0)
        step_text.set_text("Step: Prediction")
        ax.set_title("1. Prediction: Motion update")
        color = 'gray'

    elif current_step == 'sense':
        # 2. Weight update based on sensing
        for p in particles:
            p.weight = sense(p, robot_position)
        step_text.set_text("Step: Sensing / Weighting")
        ax.set_title("2. Sensing: Weight Update")
        color = 'blue'

    elif current_step == 'resample':
        # 3. Resample based on weights
        particles = resample(particles)
        step_text.set_text("Step: Resampling")
        ax.set_title("3. Resampling: Particle Selection")
        color = 'green'

    elif current_step == 'move':
        # 4. Robot moves
        robot_position[0] += MOVE_DISTANCE
        step_text.set_text("Step: Robot Moving")
        ax.set_title("4. Robot Moves")
        color = 'red'

    step_counter += 1

    # Draw
    xs = [p.x for p in particles]
    ys = [p.y for p in particles]
    ax.scatter(xs, ys, color=color, alpha=0.6, label='Particles')
    ax.scatter(robot_position[0], robot_position[1], color='red', s=100, label='Robot')
    ax.legend()
    ax.text(0.02, 0.95, f"Step: {current_step.upper()}", transform=ax.transAxes)

ani = animation.FuncAnimation(fig, update, frames=100, interval=1000)
plt.show()