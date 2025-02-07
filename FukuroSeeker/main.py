import numpy as np
import pygame
import math

# mimic another package in fukurobot
class FukuroBot:
    def __init__(self,
                 pos_x: float = 0.0, 
                 pos_y: float = 0.0,
                 radius: float = 4.0,
                 field_width: float = 800.0,
                 field_height: float = 600.0):
        self.position = np.array([pos_x, pos_y], dtype=float)
        self.radius = radius
        
        self.field_width = field_width
        self.field_height = field_height
        self.path = None
        
    def set_path(self, 
                 path: np.array = None,
                 n_points: int = 7,
                 random: bool = True):
        if random:
            self.path = np.random.randint(0, [self.field_width, self.field_height], size=(n_points, 2))
        else:
            self.path = path

    # ball behaviour 
    def set_ball(self):
        pass


# seeker class
class FukuroSeeker:
    def __init__(self):
        self.position = None
        self.velocity = np.array([0, 0], dtype=float)
        self.acceleration = np.array([0, 0], dtype=float)
        self.steer = np.array([0, 0], dtype=float)
        self.maxspeed = 1.15
        self.maxforce = 0.3 # Kp
        self.linear_tolerance = 15
        self.angular_tolerance = 10
        self.slowdown_tolerance = 200
        self.mass = 30.0 # 1.0  # Mass of the robot
        self.friction_coefficient = 0.002 # 0.02  # Friction coefficient
        self.drag_coefficient = 0.005 # 0.005  # Drag coefficient

        self.path = None
        self.path_length = None
        self.path_idx = None

        self.ball = None

        self.target_pos = np.array([0, 0], dtype=float)
        self.target_distance = None
        self.target_angle = 0

        self.angular_velocity = 0
        self.angle_before = 0

        self.motor_speed = None

    # path following
    def seek(self):
        # calculate desired velocity
        desired = self.target - self.position
        self.target_distance = self.calculate_distance(desired)

        if self.target_distance <= self.slowdown_tolerance:
            desired = self.set_mag(desired, self.maxspeed * (self.target_distance / 100))
        else:
            desired = self.set_mag(desired, self.maxspeed)

        self.steer = desired - self.velocity
        self.steer = self.limit(self.steer, self.maxforce)

        # apply physic forces
        friction = -self.friction_coefficient * self.velocity
        drag = -self.drag_coefficient * np.square(self.velocity)

        net_force = self.steer + friction + drag # net force calculation

        # update acceleration based on force and mass
        self.acceleration += net_force / self.mass

        # update velocity and position
        self.velocity += self.acceleration
        self.velocity = self.limit(self.velocity, self.maxspeed)
        self.position += self.velocity
        self.acceleration *= 0  # reset acceleration after applying it

        # calculate angle difference
        self.target_angle = self.calculate_angle(self.velocity, self.target-self.position)

        # Update angular velocity   
        self.angular_velocity = self.target_angle - self.angle_before
        self.angle_before = self.target_angle
        # self.angular_velocity = self.target_angle

        self.set_motor_speed()

    # interception controller
    def seek_interception(self, kp=0.5, kd=0.2):
        desired = self.ball[0] - self.position[0] # horizontal error
    
        desired_derivative = -self.velocity[0] # horizontal rate of change

        control_force = kp * desired + kd * desired_derivative
        # control_force = np.clip(control_force, -self.maxforce, self.maxforce)

        self.steer = np.array([control_force, 0]) - self.velocity
        self.steer = self.limit(self.steer, self.maxforce)

        # apply physic forces
        friction = -self.friction_coefficient * self.velocity
        drag = -self.drag_coefficient * np.square(self.velocity)

        net_force = self.steer + friction + drag # net force calculation

        # update acceleration based on force and mass
        self.acceleration += net_force / self.mass

        # update velocity and position
        self.velocity += self.acceleration
        self.velocity = self.limit(self.velocity, self.maxspeed)
        self.position += self.velocity
        self.acceleration *= 0  # reset acceleration after applying it

        # calculate angle difference
        self.target_angle = self.calculate_angle(self.velocity, self.target-self.position)

        # Update angular velocity   
        self.angular_velocity = self.target_angle - self.angle_before
        self.angle_before = self.target_angle
        # self.angular_velocity = self.target_angle

        self.set_motor_speed()

    def set_motor_speed(self):
        converter = np.array([[math.cos(math.radians(30)), math.sin(math.radians(30)), 0.185],
                              [-math.cos(math.radians(30)), math.sin(math.radians(30)), 0.185],
                              [0                , -1              , 0.185]])

        # convert global speed into local speed
        # watch the negative sign
        angle  = self.calculate_angle(np.array([0, 0]), self.velocity)
        angle = np.radians(-angle)

        local_x = math.cos(-angle) * self.velocity[0] - math.sin(-angle) * self.velocity[1]
        local_y = math.cos(-angle) * self.velocity[1] + math.sin(-angle) * self.velocity[0]

        local_vel = np.array([local_x, local_y])
        velocity = np.append(local_vel, self.angular_velocity) # tiba tiba speed motor gede banget
        velocity = velocity.reshape((3, 1))
        self.motor_speed = converter @ velocity
        self.motor_speed = self.motor_speed.reshape(-1,)

        for i in range(len(self.motor_speed)):
            if self.motor_speed[i] > self.maxspeed:
                self.motor_speed[i] = self.maxspeed
            
    def get_ball(self, ball: np.array):
        self.ball = ball

    def get_path(self, path: np.array):
        self.path = path
        self.path_length = self.path.shape[0] # - 1 
        self.path_idx = 0
        self.target = self.path[self.path_idx]

    def update_position(self, position):
        self.position = position
    
    def update_target(self):
        self.target = self.path[self.path_idx]

    @staticmethod
    def limit(vector, max_value):
        mag = np.linalg.norm(vector)
        if mag > max_value:
            return (vector / mag) * max_value
        return vector

    @staticmethod
    def set_mag(vector, magnitude):
        mag = np.linalg.norm(vector)
        if mag != 0:
            return (vector / mag) * magnitude
        return vector

    @staticmethod
    def calculate_distance(desired) -> float:
        return np.linalg.norm(desired)

    @staticmethod
    def calculate_angle(velocity, target) -> float:
        velocity = np.arctan2(velocity[1], velocity[0])
        target = np.arctan2(target[1], target[0])

        return np.degrees(velocity - target)

# pygame simulator
class Simulator:
    def __init__(self,
                 robot, 
                 seeker,
                 width: float = 800, 
                 height: float = 600):
        pygame.init()
        self.WIDTH = width
        self.HEIGHT = height
        self.SCREEN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Fukuro Seek Algorithm Demo")

        self.CLOCK = pygame.time.Clock()
        self.FPS = 60
        self.RUNNING = True

        self.robot = robot
        self.seeker = seeker

    def draw_path(self,
                  path):
        for pos in path:
            pygame.draw.circle(self.SCREEN, (255, 0, 0), tuple(pos.astype(int)), 3)
        for i in range(1, len(path)):
            pygame.draw.line(self.SCREEN, (0, 255, 0), tuple(path[i - 1].astype(int)), tuple(path[i].astype(int)), 2)

    def draw_robot(self,
                   pos_x, 
                   pos_y,
                   radius, 
                   velocity):
        theta = np.arctan2(velocity[1], velocity[0]) # / (np.pi / 2)
        points = [
            (pos_x + radius * np.cos(theta), pos_y + radius * np.sin(theta)),
            (pos_x + radius * np.cos(theta + 2 * np.pi / 3), pos_y + radius * np.sin(theta + 2 * np.pi / 3)),
            (pos_x + radius * np.cos(theta - 2 * np.pi / 3), pos_y + radius * np.sin(theta - 2 * np.pi / 3)),
        ]

        pygame.draw.polygon(self.SCREEN, (127, 127, 127), points)
        pygame.draw.polygon(self.SCREEN, (0, 0, 0), points, 1)

    def run_simulation(self):

        while self.RUNNING:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.RUNNING = False
            
            self.seeker.update_position(self.robot.position)
            self.seeker.seek()
            
            if self.seeker.path_idx != self.seeker.path_length - 1 and self.seeker.target_distance < self.seeker.linear_tolerance:
                self.seeker.path_idx += 1
                self.seeker.update_target()

            self.SCREEN.fill((255, 255, 255))

            self.draw_path(self.robot.path)
            self.draw_robot(self.robot.position[0], 
                            self.robot.position[1], 
                            self.robot.radius, 
                            self.seeker.velocity)
            self.generate_velocity_monitor()
            self.generate_motor_monitor()

            pygame.display.flip()
            self.CLOCK.tick(self.FPS)
        
        pygame.quit()

    def generate_velocity_monitor(self):
        pygame.draw.line(self.SCREEN, (0, 125, 255), 
                         tuple(self.robot.position.astype(int)), 
                         tuple((self.robot.position + self.seeker.velocity*50).astype(int)), 
                         2)

    def generate_motor_monitor(self):
        font = pygame.font.Font(None, 36)

        text_color = (255, 0, 0)  
        x_margin = 10
        y_margin = 10
        y_offset = y_margin 

        display_motor = self.seeker.motor_speed
        display_motor = display_motor.tolist()
        display_motor.append(self.seeker.target_angle)
   
        for i, speed in enumerate(display_motor): 
            if i < 3:
                text = f"Motor {i + 1}: {speed:.5f}"
            else:
                text = f"Angle difference: {speed:.2f}"

            text_surface = font.render(text, True, text_color)
            text_rect = text_surface.get_rect()

            text_rect.topleft = (x_margin, y_offset)
            self.SCREEN.blit(text_surface, text_rect)

            y_offset += text_surface.get_height() + 5 

if __name__ == "__main__":

    path = np.array([
        [100, 500],
        [700, 500],
        [100, 100],
        [700, 100],
    ])

    path2 = np.array([
        [700, 100],
        [100, 100],
        [700, 500],
        [100, 500],
    ])

    robot = FukuroBot()

    robot.set_path(path=path2, random=True)

    seeker = FukuroSeeker()
    seeker.get_path(robot.path)

    simulator = Simulator(robot, seeker)
    simulator.run_simulation()