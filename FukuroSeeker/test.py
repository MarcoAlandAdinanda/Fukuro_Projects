import pygame
import threading
import sys

def create_window(x, y, width, height):
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.NOFRAME, display=1)
    pygame.display.set_caption(f"Window at ({x}, {y})")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill((0, 0, 255))  # Fill the window with a blue color
        pygame.display.flip()
    
    pygame.quit()

def main():
    pygame.init()

    # Create two threads for two windows
    window1 = threading.Thread(target=create_window, args=(0, 0, 640, 480))
    window2 = threading.Thread(target=create_window, args=(640, 0, 640, 480))
    
    window1.start()
    window2.start()

    window1.join()
    window2.join()

    pygame.quit()

if __name__ == "__main__":
    main()
