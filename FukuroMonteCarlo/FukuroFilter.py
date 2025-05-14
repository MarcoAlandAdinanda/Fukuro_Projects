import numpy as np
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

class Map:
    def __init__(self):
        self.width = 10
        self.height = 10
        self.obstacles = [
            {'x': 2, 'y': 2, 'w': 1, 'h': 1},
            {'x': 7, 'y': 7, 'w': 1, 'h': 1}
        ]
    
    def is_occupied(self, x, y):
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return True
        for obs in self.obstacles:
            if (obs['x'] <= x <= obs['x'] + obs['w'] and
                obs['y'] <= y <= obs['y'] + obs['h']):
                return True
        return False

class AMCL:
    def __init__(self, map, num_particles=125):
        self.map = map
        self.particles = []
        self.num_particles = num_particles
        for _ in range(num_particles):
            x = np.random.uniform(0, map.width)
            y = np.random.uniform(0, map.height)
            theta = np.random.uniform(0, 2*np.pi)
            self.particles.append(Particle(x, y, theta, 1.0/num_particles))
    
    def predict(self, dx, dtheta):
        for p in self.particles:
            # Add motion noise
            noisy_dx = dx + np.random.normal(0, 0.1)
            noisy_dtheta = dtheta + np.random.normal(0, 0.05)
            
            # Update pose
            p.x += noisy_dx * np.cos(p.theta)
            p.y += noisy_dx * np.sin(p.theta)
            p.theta += noisy_dtheta
            p.theta %= 2*np.pi
    
    def update(self, sensor_readings, sensor_std=0.2):
        for p in self.particles:
            simulated = self.simulate_sensor(p)
            likelihood = 1.0
            for real, sim in zip(sensor_readings, simulated):
                likelihood *= np.exp(-(real - sim)**2 / (2 * sensor_std**2))
            p.weight = likelihood
        
        # Normalize weights
        total = sum(p.weight for p in self.particles)
        if total == 0:
            for p in self.particles:
                p.weight = 1.0/self.num_particles
        else:
            for p in self.particles:
                p.weight /= total
    
    def simulate_sensor(self, particle):
        readings = []
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            world_angle = particle.theta + angle
            x, y = particle.x, particle.y
            dx = np.cos(world_angle) * 0.1
            dy = np.sin(world_angle) * 0.1
            distance = 0.0
            
            for _ in range(50):
                x += dx
                y += dy
                if self.map.is_occupied(x, y):
                    break
                distance += 0.1
            readings.append(min(distance, 5.0))
        return readings
    
    def resample(self):
        weights = [p.weight for p in self.particles]
        cumulative = np.cumsum(weights)
        new_particles = []
        
        step = 1.0/self.num_particles
        pointer = np.random.uniform(0, step)
        
        for _ in range(self.num_particles):
            idx = np.searchsorted(cumulative, pointer)
            idx = min(idx, len(self.particles)-1)
            
            # Add noise to prevent particle deprivation
            new_x = self.particles[idx].x + np.random.normal(0, 0.02)
            new_y = self.particles[idx].y + np.random.normal(0, 0.02)
            new_theta = self.particles[idx].theta + np.random.normal(0, 0.01)
            new_theta %= 2*np.pi
            
            new_particles.append(Particle(new_x, new_y, new_theta, 1.0/self.num_particles))
            pointer += step
        
        self.particles = new_particles
    
    def estimate_pose(self):
        x = sum(p.x * p.weight for p in self.particles)
        y = sum(p.y * p.weight for p in self.particles)
        
        # Circular mean for theta
        vectors = [(np.cos(p.theta), np.sin(p.theta)) for p in self.particles]
        mean_vec = np.sum([v[0]*p.weight for v, p in zip(vectors, self.particles)]), \
                   np.sum([v[1]*p.weight for v, p in zip(vectors, self.particles)])
        theta = np.arctan2(mean_vec[1], mean_vec[0])
        return (x, y, theta)

# Simulation
map = Map()
amcl = AMCL(map, 1000)

# True robot state
true_pose = {'x': 5.0, 'y': 5.0, 'theta': 0.0}
estimated = []
actual = []

for _ in range(50):
    # Move robot
    true_pose['x'] += 0.2 * np.cos(true_pose['theta']) + np.random.normal(0, 0.02)
    true_pose['y'] += 0.2 * np.sin(true_pose['theta']) + np.random.normal(0, 0.02)
    true_pose['theta'] += np.random.normal(0, 0.01)
    true_pose['theta'] %= 2*np.pi
    
    # Generate sensor readings
    sensor_data = []
    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
        world_angle = true_pose['theta'] + angle
        x, y = true_pose['x'], true_pose['y']
        dx = np.cos(world_angle) * 0.1
        dy = np.sin(world_angle) * 0.1
        dist = 0.0
        
        for _ in range(50):
            x += dx
            y += dy
            if map.is_occupied(x, y):
                break
            dist += 0.1
        sensor_data.append(min(dist, 5.0))
    
    # AMCL steps
    amcl.predict(0.2, 0.0)
    amcl.update(sensor_data)
    amcl.resample()
    
    # Store results
    est = amcl.estimate_pose()
    estimated.append((est[0], est[1]))
    actual.append((true_pose['x'], true_pose['y']))

# Plot results
estimated = np.array(estimated)
actual = np.array(actual)

plt.figure(figsize=(10, 6))
plt.plot(actual[:,0], actual[:,1], 'g-', label='True Path')
plt.plot(estimated[:,0], estimated[:,1], 'r--', label='Estimated Path')
plt.legend()
plt.title('AMCL Localization Results')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.grid(True)
plt.show()