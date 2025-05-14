import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import math

class ParticleFilter:
    def __init__(self, num_particles, map_size, landmarks):
        self.num_particles = num_particles
        self.map_width, self.map_height = map_size
        self.landmarks = landmarks
        
        # Initialize particles
        self.particles = []
        for _ in range(num_particles):
            x = random.uniform(0, map_size[0])
            y = random.uniform(0, map_size[1])
            theta = random.uniform(0, 2 * math.pi)
            weight = 1.0 / num_particles
            self.particles.append({'x': x, 'y': y, 'theta': theta, 'weight': weight})
    
    def predict(self, motion):
        for p in self.particles:
            # Add noise to motion
            noisy_dx = motion['dx'] + random.gauss(0, 0.1)
            noisy_dy = motion['dy'] + random.gauss(0, 0.1)
            noisy_dtheta = motion['dtheta'] + random.gauss(0, 0.05)
            
            # Update particle position
            p['x'] += noisy_dx * math.cos(p['theta']) - noisy_dy * math.sin(p['theta'])
            p['y'] += noisy_dx * math.sin(p['theta']) + noisy_dy * math.cos(p['theta'])
            p['theta'] += noisy_dtheta
            p['theta'] %= 2 * math.pi
            
            # Keep particles within map bounds
            p['x'] = max(0, min(self.map_width, p['x']))
            p['y'] = max(0, min(self.map_height, p['y']))
    
    def update(self, measurements):
        for p in self.particles:
            expected_measurements = []
            for lm in self.landmarks:
                dx = lm[0] - p['x']
                dy = lm[1] - p['y']
                dist = math.sqrt(dx**2 + dy**2)
                expected_measurements.append(dist)
            
            weight = 1.0
            for meas, exp in zip(measurements, expected_measurements):
                prob = math.exp(-(meas - exp)**2 / (2 * 0.5**2))
                weight *= prob
            
            p['weight'] = weight
        
        # Normalize weights
        total_weight = sum(p['weight'] for p in self.particles)
        if total_weight > 0:
            for p in self.particles:
                p['weight'] /= total_weight
    
    def resample(self):
        new_particles = []
        index = int(random.random() * self.num_particles)
        beta = 0.0
        max_weight = max(p['weight'] for p in self.particles)
        
        for _ in range(self.num_particles):
            beta += random.random() * 2.0 * max_weight
            while beta > self.particles[index]['weight']:
                beta -= self.particles[index]['weight']
                index = (index + 1) % self.num_particles
            new_particles.append({
                'x': self.particles[index]['x'],
                'y': self.particles[index]['y'],
                'theta': self.particles[index]['theta'],
                'weight': 1.0 / self.num_particles
            })
        
        self.particles = new_particles
    
    def get_estimate(self):
        x = sum(p['x'] * p['weight'] for p in self.particles)
        y = sum(p['y'] * p['weight'] for p in self.particles)
        
        sin_sum = sum(math.sin(p['theta']) * p['weight'] for p in self.particles)
        cos_sum = sum(math.cos(p['theta']) * p['weight'] for p in self.particles)
        theta = math.atan2(sin_sum, cos_sum)
        
        return x, y, theta


# Simulation setup
map_size = (100, 100)
landmarks = [(20, 20), (80, 20), (20, 80), (80, 80)]
pf = ParticleFilter(num_particles=1000, map_size=map_size, landmarks=landmarks)

# True robot state (unknown to filter)
true_x, true_y, true_theta = 30, 30, math.pi/4

# Create figure for animation
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, map_size[0])
ax.set_ylim(0, map_size[1])
ax.grid(True)

# Create plot elements
landmark_plot = ax.plot([], [], 'ro', markersize=10, label='Landmarks')[0]
particle_plot = ax.scatter([], [], c='b', alpha=0.2, s=5, label='Particles')
estimate_plot = ax.plot([], [], 'go', markersize=10, label='Estimate')[0]
true_plot = ax.plot([], [], 'kx', markersize=15, label='True Position')[0]
ax.legend()

# Text annotation for status
status_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    """Initialize animation"""
    landmark_plot.set_data([lm[0] for lm in landmarks], [lm[1] for lm in landmarks])
    particle_plot.set_offsets(np.empty((0, 2)))
    estimate_plot.set_data([], [])
    true_plot.set_data([], [])
    status_text.set_text('')
    return landmark_plot, particle_plot, estimate_plot, true_plot, status_text

def update(frame):
    """Update function for animation"""
    global true_x, true_y, true_theta
    
    # Robot moves forward 5 units
    motion = {'dx': 5, 'dy': 0, 'dtheta': 0}
    
    # Update true position
    true_x += motion['dx'] * math.cos(true_theta) - motion['dy'] * math.sin(true_theta)
    true_y += motion['dx'] * math.sin(true_theta) + motion['dy'] * math.cos(true_theta)
    
    # Keep true position within bounds
    true_x = max(0, min(map_size[0], true_x))
    true_y = max(0, min(map_size[1], true_y))
    
    # Get true measurements (with noise)
    measurements = []
    for lm in landmarks:
        dx = lm[0] - true_x
        dy = lm[1] - true_y
        dist = math.sqrt(dx**2 + dy**2) + random.gauss(0, 0.3)
        measurements.append(dist)
    
    # Update particle filter
    pf.predict(motion)
    pf.update(measurements)
    pf.resample()
    
    # Get estimate
    est_x, est_y, est_theta = pf.get_estimate()
    
    # Update plots
    particle_plot.set_offsets([(p['x'], p['y']) for p in pf.particles])
    estimate_plot.set_data(est_x, est_y)
    true_plot.set_data(true_x, true_y)
    
    # Update status text
    status_text.set_text(f'Step: {frame}\nTrue: ({true_x:.1f}, {true_y:.1f})\nEstimate: ({est_x:.1f}, {est_y:.1f})')
    
    return landmark_plot, particle_plot, estimate_plot, true_plot, status_text

# Create animation
ani = FuncAnimation(
    fig, 
    update, 
    frames=20, 
    init_func=init, 
    blit=True, 
    interval=500,  # 0.5 second between frames
    repeat=False
)

plt.title('Monte Carlo Robot Localization')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.show()

# To save the animation (requires ffmpeg):
# ani.save('robot_localization.mp4', writer='ffmpeg', fps=2, dpi=300)