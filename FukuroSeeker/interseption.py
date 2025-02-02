import pygame
import numpy as np

# Define the Vehicle class
class Vehicle:
    def __init__(self, x, y, maxspeed=6, maxforce=0.2):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([0, 0], dtype=float)
        self.acceleration = np.array([0, 0], dtype=float)
        self.r = 6
        self.maxspeed = maxspeed
        self.maxforce = maxforce

    def update(self):
        self.velocity += self.acceleration
        self.velocity = self.limit(self.velocity, self.maxspeed)
        self.position += self.velocity
        self.acceleration *= 0

    def apply_force(self, force):
        self.acceleration += force

    def seek_horizontal(self, target_x, kp=0.5, kd=0.2):
        # Error (distance to target x position)
        error = target_x - self.position[0]

        # Derivative (rate of change of error)
        error_derivative = -self.velocity[0]

        # PD control to calculate force
        control_force = kp * error + kd * error_derivative
        control_force = np.clip(control_force, -self.maxforce, self.maxforce)

        # Apply force in the x direction only
        self.apply_force(np.array([control_force, 0]))

    def display(self, screen, color=(127, 127, 127)):
        theta = np.arctan2(self.velocity[1], self.velocity[0]) + np.pi / 2
        x, y = self.position

        # Draw the triangle representing the vehicle
        points = [
            (x + self.r * np.cos(theta), y + self.r * np.sin(theta)),
            (x + self.r * np.cos(theta + 2 * np.pi / 3), y + self.r * np.sin(theta + 2 * np.pi / 3)),
            (x + self.r * np.cos(theta - 2 * np.pi / 3), y + self.r * np.sin(theta - 2 * np.pi / 3)),
        ]

        pygame.draw.polygon(screen, color, points)
        pygame.draw.polygon(screen, (0, 0, 0), points, 1)

    @staticmethod
    def limit(vector, max_value):
        mag = np.linalg.norm(vector)
        if mag > max_value:
            return (vector / mag) * max_value
        return vector

# Initialize pygame
pygame.init()

# Screen settings
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PD Control Goalkeeper")
clock = pygame.time.Clock()

# Create the goalkeeper robot (horizontal movement only)
goalkeeper = Vehicle(WIDTH // 2, HEIGHT - 50)  # Near the bottom of the screen

# Create the incoming robot (attacker)
attacker = Vehicle(WIDTH // 4, HEIGHT // 4)  # Start near the top-left corner
attacker.velocity = np.array([3, 2], dtype=float)  # Initial velocity for the attacker

# Prediction time (seconds)
prediction_time = 1.0

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the attacker robot's position
    attacker.position += attacker.velocity
    # Bounce the attacker off screen edges
    if attacker.position[0] < 0 or attacker.position[0] > WIDTH:
        attacker.velocity[0] *= -1
    if attacker.position[1] < 0 or attacker.position[1] > HEIGHT:
        attacker.velocity[1] *= -1

    # Predict the attacker's future x position
    future_x = attacker.position[0] + attacker.velocity[0] * prediction_time

    # Goalkeeper logic: PD control to intercept the predicted x position
    goalkeeper.seek_horizontal(future_x)

    # Update and display both robots
    screen.fill((255, 255, 255))
    goalkeeper.update()
    goalkeeper.display(screen, color=(255, 0, 0))  # Red goalkeeper
    attacker.display(screen, color=(0, 0, 255))  # Blue attacker

    # Display instructions
    font = pygame.font.SysFont(None, 36)
    instruction_text = "Red: Goalkeeper (PD Control) | Blue: Attacker"
    text_surface = font.render(instruction_text, True, (0, 0, 0))
    screen.blit(text_surface, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
