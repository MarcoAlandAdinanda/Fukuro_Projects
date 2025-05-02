import pygame
import numpy as np
import random

# Inisialisasi pygame
pygame.init()
width, height = 600, 400
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Monte Carlo Localization with Lines")

# Warna
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY  = (100, 100, 100)

# Jumlah partikel
NUM_PARTICLES = 500*2

# Partikel
class Particle:
    def __init__(self):
        self.x = random.uniform(0, width)
        self.y = random.uniform(0, height)
        self.weight = 1.0

    def move(self, dx, dy):
        self.x += dx + np.random.normal(0, 2)
        self.y += dy + np.random.normal(0, 2)
        self.x = max(0, min(width, self.x))
        self.y = max(0, min(height, self.y))

    def sense(self):
        return [
            self.x,
            width - self.x,
            self.y,
            height - self.y
        ]

# Robot
class Robot:
    def __init__(self):
        self.x = width / 2
        self.y = height / 2

    def move(self, dx, dy):
        self.x += dx
        self.y += dy
        self.x = max(0, min(width, self.x))
        self.y = max(0, min(height, self.y))

    def sense(self):
        return [
            self.x,
            width - self.x,
            self.y,
            height - self.y
        ]

# Resampling
def resample(particles):
    weights = [p.weight for p in particles]
    total = sum(weights)
    if total == 0:
        return [Particle() for _ in particles]
    probs = [w / total for w in weights]
    indices = np.random.choice(len(particles), len(particles), p=probs)
    return [particles[i] for i in indices]

# Fungsi utama
def main():
    clock = pygame.time.Clock()
    robot = Robot()
    particles = [Particle() for _ in range(NUM_PARTICLES)]

    running = True
    while running:
        clock.tick(30)
        win.fill(BLACK)

        dx = dy = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Input kontrol gerak
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: dx = -2
        if keys[pygame.K_RIGHT]: dx = 2
        if keys[pygame.K_UP]: dy = -2
        if keys[pygame.K_DOWN]: dy = 2

        robot.move(dx, dy)
        real_sense = robot.sense()

        # Update partikel
        for p in particles:
            p.move(dx, dy)
            p_sense = p.sense()
            error = np.sum([(a - b) ** 2 for a, b in zip(real_sense, p_sense)])
            p.weight = np.exp(-error / 5000)

        # Resampling
        particles = resample(particles)

        # Gambar garis dan partikel
        for p in particles:
            pygame.draw.line(win, GREEN, (int(p.x), int(p.y)), (int(robot.x), int(robot.y)), 1)
            pygame.draw.circle(win, WHITE, (int(p.x), int(p.y)), 2)

        # Gambar robot
        pygame.draw.circle(win, RED, (int(robot.x), int(robot.y)), 5)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
