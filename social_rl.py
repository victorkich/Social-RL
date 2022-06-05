import numpy as np
import pygame
import time
pygame.init()


def grab_event():
    pass


def release_event():
    pass


x_boundarie = 800
y_boundarie = 800
agent_radius = 25

x_jobs = [50, 50, 750, 750]
y_jobs = [50, 750, 50, 750]
job_radius = 50

x_banks = [300, 300, 500, 500]
y_banks = [300, 500, 300, 500]
bank_radius = 40

x_locks = [150, 400, 400, 650]
y_locks = [400, 150, 650, 400]
lock_radius = 30

scr = pygame.display.set_mode((x_boundarie, y_boundarie)) 
pygame.display.set_caption('Social RL') 
x_boundarie -= agent_radius
y_boundarie -= agent_radius

agent_x = 250
agent_y = 250
running = True

frame_cap = 1.0/60
time_1 = time.perf_counter()
unprocessed = 0

while running:
    can_render = False
    time_2 = time.perf_counter()
    passed = time_2 - time_1
    unprocessed += passed
    time_1 = time_2

    while unprocessed >= frame_cap:
        unprocessed -= frame_cap
        can_render = True

    if can_render:
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                running = False

            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_w:
                        agent_y -= 10
                    case pygame.K_a:
                        agent_x -= 10
                    case pygame.K_s:
                        agent_y += 10
                    case pygame.K_d:
                        agent_x += 10
                    case pygame.K_e:
                        grab_event()
                    case pygame.K_r:
                        release_event()

        agent_x += np.random.randn()
        agent_y += np.random.randn()

        if agent_x > x_boundarie:
            agent_x = x_boundarie
        elif agent_x < agent_radius:
            agent_x = agent_radius
        if agent_y > y_boundarie:
            agent_y = y_boundarie
        elif agent_y < agent_radius:
            agent_y = agent_radius
        
        scr.fill((0, 0, 0))
        for i in range(4):
            pygame.draw.circle(scr, (0, 0, 200), (x_banks[i], y_banks[i]), bank_radius)
            pygame.draw.circle(scr, (0, 200, 0), (x_jobs[i], y_jobs[i]), job_radius)
            pygame.draw.circle(scr, (200, 200, 200), (x_locks[i], y_locks[i]), lock_radius)

        pygame.draw.circle(scr, (200, 0, 200), (agent_x, agent_y), agent_radius)
        pygame.display.flip()
    
pygame.quit() 
