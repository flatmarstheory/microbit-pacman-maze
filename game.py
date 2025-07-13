import pygame
import threading
import time
import math
import random
import numpy as np
from kaspersmicrobit import KaspersMicrobit
from kaspersmicrobit.services.accelerometer import AccelerometerData
from kaspersmicrobit.services.io_pin import (
    Pin, PinADConfiguration, PinIOConfiguration,
    PinAD, PinIO, PwmControlData
)

# --- Pygame Setup ---
pygame.init()
TILE_SIZE = 48
GRID_COLS, GRID_ROWS = 24, 14
WIDTH, HEIGHT = GRID_COLS * TILE_SIZE, GRID_ROWS * TILE_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man Mango Maze: Enhanced Q-Learning AI")
clock = pygame.time.Clock()
font_main = pygame.font.SysFont(None, 64)
font_small = pygame.font.SysFont(None, 24)
running = True

# --- Maze Layout ---
maze = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        if r == 0 or r == GRID_ROWS-1 or c == 0 or c == GRID_COLS-1:
            maze[r][c] = 1
for c in range(1, GRID_COLS-1):
    if 3 <= c <= GRID_COLS-4:
        maze[4][c] = maze[10][c] = 1
for r in range(1, GRID_ROWS-1):
    if 5 <= r <= GRID_ROWS-6:
        maze[r][8] = maze[r][15] = 1

# --- RL Parameters ---
actions = [(1,0), (-1,0), (0,1), (0,-1)]  # right, left, down, up
num_actions = len(actions)
Q = np.zeros((GRID_ROWS, GRID_COLS, num_actions), dtype=np.float32)
alpha = 0.7  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.5  # exploration rate
min_epsilon = 0.1
decay_rate = 0.995  # epsilon decay
timestep_penalty = -0.01  # penalty each move
wall_penalty = -0.5   # penalty for hitting wall

# --- Game State ---
score = 0
player = [GRID_COLS//2, GRID_ROWS//2]
raw_input = [0.0, 0.0]
target_pixel = pygame.Vector2(player[0]*TILE_SIZE+TILE_SIZE/2,
                              player[1]*TILE_SIZE+TILE_SIZE/2)
pixel_pos = target_pixel.copy()
move_delay = 80
last_move = pygame.time.get_ticks()

# --- Mango Setup ---
MANGO_SIZE, MANGO_RADIUS = 60, 30
MAX_MANGOES = 5
mangoes = []
mango_img = pygame.image.load("mango.png").convert_alpha()
mango_img = pygame.transform.scale(mango_img, (MANGO_SIZE, MANGO_SIZE))

# --- Micro:bit Setup ---
microbit_device = None

def accelerometer_data(d: AccelerometerData):
    raw_input[0] = d.x / 100.0
    raw_input[1] = -d.y / 100.0

def microbit_thread():
    global microbit_device
    with KaspersMicrobit.find_one_microbit() as mb:
        microbit_device = mb
        # configure P0 for PWM sound
        ad_map = [PinAD.ANALOG if p == Pin.P0 else PinAD.DIGITAL for p in Pin]
        mb.io_pin.write_ad_configuration(PinADConfiguration(ad_map))
        io_map = [PinIO.OUTPUT if p == Pin.P0 else PinIO.INPUT for p in Pin]
        mb.io_pin.write_io_configuration(PinIOConfiguration(io_map))
        mb.accelerometer.notify(accelerometer_data)
        while running:
            time.sleep(0.05)

threading.Thread(target=microbit_thread, daemon=True).start()

# --- Sound on Eat ---
def play_sound():
    if microbit_device:
        period = int(1e6/880)
        tone = PwmControlData(Pin.P0, 512, period)
        microbit_device.io_pin.write_pwm_control_data(tone)
        time.sleep(0.1)
        microbit_device.io_pin.write_pwm_control_data(PwmControlData(Pin.P0, 0, period))

# --- Helpers ---
def can_move(c, r):
    return 0 <= c < GRID_COLS and 0 <= r < GRID_ROWS and maze[r][c] == 0

def spawn_mango():
    while True:
        c = random.randint(1, GRID_COLS-2)
        r = random.randint(1, GRID_ROWS-2)
        if maze[r][c] == 0 and [c, r] not in mangoes and [c, r] != player:
            return [c, r]

# --- Main Loop ---
frame = 0
while running:
    dt = clock.tick(60)
    now = pygame.time.get_ticks()
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    # State Q-values
    pr, pc = player[1], player[0]
    state_Q = Q[pr, pc]

    # Determine human tilt override
    dx, dy = raw_input
    human_act = None
    if math.hypot(dx, dy) > 0.5:
        if abs(dx) > abs(dy):
            human_act = 0 if dx > 0 else 1
        else:
            human_act = 2 if dy > 0 else 3

    # Select action: priority to human, else epsilon-greedy AI
    if human_act is not None:
        act = human_act
    else:
        if random.random() < epsilon:
            act = random.randrange(num_actions)
        else:
            act = int(np.argmax(state_Q))
        # decay epsilon
        epsilon = max(min_epsilon, epsilon * decay_rate)

    # Movement & Q-learning update
    if now - last_move > move_delay:
        last_move = now
        r0, c0 = pr, pc
        dc, dr = actions[act]
        nc, nr = c0 + dc, r0 + dr
        # Initialize reward
        reward = timestep_penalty
        # Check wall hit
        if not can_move(nc, nr):
            reward = wall_penalty
        else:
            # Move
            player = [nc, nr]
            # Check eating
            for i in range(len(mangoes)-1, -1, -1):
                m = mangoes[i]
                if m[0] == player[0] and m[1] == player[1]:
                    mangoes.pop(i)
                    score += 1
                    reward = 1.0
                    threading.Thread(target=play_sound, daemon=True).start()
        # Q-learning update
        r1, c1 = player[1], player[0]
        best_next = np.max(Q[r1, c1])
        Q[r0, c0, act] += alpha * (reward + gamma * best_next - Q[r0, c0, act])
        # Update pixel target
        target_pixel = pygame.Vector2(player[0]*TILE_SIZE+TILE_SIZE/2,
                                      player[1]*TILE_SIZE+TILE_SIZE/2)

    # Smooth movement
    pixel_pos += (target_pixel - pixel_pos) * min(1, dt/move_delay)

    # Spawn mangoes
    while len(mangoes) < MAX_MANGOES:
        mangoes.append(spawn_mango())

    # Draw scene
    screen.fill((0,0,0))
    for x in range(0, WIDTH, TILE_SIZE):
        pygame.draw.line(screen, (30,30,30), (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, TILE_SIZE):
        pygame.draw.line(screen, (30,30,30), (0, y), (WIDTH, y))
    for rr in range(GRID_ROWS):
        for cc in range(GRID_COLS):
            if maze[rr][cc]:
                pygame.draw.rect(screen, (0,0,128), (cc*TILE_SIZE, rr*TILE_SIZE, TILE_SIZE, TILE_SIZE))
    for m in mangoes:
        p = pygame.Vector2(m[0]*TILE_SIZE+TILE_SIZE/2, m[1]*TILE_SIZE+TILE_SIZE/2)
        screen.blit(mango_img, (p.x-MANGO_RADIUS, p.y-MANGO_RADIUS))
    # Draw Pac-Man
    mouth = 30 + 10 * math.sin(frame*0.3)
    dx, dy = actions[act]
    ang = math.degrees(math.atan2(-dy, dx))
    sa = math.radians(mouth) + math.radians(ang)
    ea = math.radians(360-mouth) + math.radians(ang)
    rect = (pixel_pos.x-TILE_SIZE/2, pixel_pos.y-TILE_SIZE/2, TILE_SIZE, TILE_SIZE)
    pygame.draw.arc(screen, (255,255,0), rect, sa, ea, TILE_SIZE//2)
    pygame.draw.polygon(screen, (0,0,0), [
        pixel_pos,
        (pixel_pos.x+(TILE_SIZE/2)*math.cos(math.radians(ang+mouth)), pixel_pos.y-(TILE_SIZE/2)*math.sin(math.radians(ang+mouth))),
        (pixel_pos.x+(TILE_SIZE/2)*math.cos(math.radians(ang-mouth)), pixel_pos.y-(TILE_SIZE/2)*math.sin(math.radians(ang-mouth)))
    ])
    # Overlays
    screen.blit(font_main.render(f"Score: {score}", True, (255,255,255)), (20,20))
    stats_y = 80
    screen.blit(font_small.render(f"Epsilon: {epsilon:.2f}", True, (200,200,200)), (20, stats_y)); stats_y += 20
    screen.blit(font_small.render(f"Alpha: {alpha}, Gamma: {gamma}", True, (200,200,200)), (20, stats_y)); stats_y += 20
    qvals = " ".join([f"Q{i}:{state_Q[i]:.2f}" for i in range(num_actions)])
    screen.blit(font_small.render(qvals, True, (200,200,200)), (20, stats_y)); stats_y += 20
    screen.blit(font_small.render("Mode: Human+AI Learning", True, (200,200,200)), (20, stats_y))

    pygame.display.flip()
    frame += 1

pygame.quit()
