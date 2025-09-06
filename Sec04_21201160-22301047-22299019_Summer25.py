from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import random
from collections import deque

# Grid constants
CELL_SIZE = 40
MAZE_WIDTH = 19
MAZE_HEIGHT = 21
WALL_HEIGHT = 50
PELLET_SIZE = 5
POWER_PELLET_SIZE = 10
GHOST_SIZE = 15
PACMAN_SIZE = 15
# Durations in milliseconds (time-based for robustness)
POWER_MODE_MS = 5000   
IFRAME_MS = 5000        
RESP_IMMUNE_MS = 5000  

# Ghost movement tick (independent of player movement)
GHOST_TICK_MS = 500  

# Pac-Man movement cooldown (ms); increase to slow player.
PACMAN_MOVE_COOLDOWN_MS = 100
pacman_last_move_ms = 0


last_ghost_step_ms = 0


# Camera and view settings

camera_mode = "2D" 
fovY = 90


# Game state featues

game_paused = False
game_over = False
game_won = False
game_score = 0
lives = 3


power_mode_until = 0  
iframe_until = 0        


# Pacman
pacman_pos = [1, 1]  
pacman_dir = [0, 0]
pacman_angle = 0
pacman_mouth = 0.0
pacman_mouth_dir = 1


# Ghost 
from collections import deque as _deque
class Ghost:
    def __init__(self, x, y, color, name):
        self.start_x = x
        self.start_y = y
        self.x = x
        self.y = y
        self.color = color
        self.name = name
        self.dir = [0, 0]
        self.eaten = False
        self.respawn_timer = 0     
        self.respawned_immune = False
        self.immune_until = 0
        # Per-ghost RNG to avoid converging paths
        self.recent = _deque(maxlen=6)  # avoid oscillations/backtracking
        self.rng = random.Random(f"{x}:{y}:{name}")

    def respawn(self):
        self.x = self.start_x
        self.y = self.start_y
        self.eaten = False
        self.respawn_timer = 0
        # Immune for 5 seconds after respawn (regardless of power mode)
        self.respawned_immune = True
        self.immune_until = glutGet(GLUT_ELAPSED_TIME) + RESP_IMMUNE_MS

ghosts = [
    Ghost(9, 9, (1, 0, 0), "Blinky"),  #orginal names 
    Ghost(8, 9, (1, 0.5, 0.5), "Pinky"),
    Ghost(10, 9, (0, 1, 1), "Inky"),
    Ghost(9, 8, (1, 0.5, 0), "Clyde")
]

maze = [
    [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1],  
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],  
    [1,2,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,2,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1],
    [1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,1,1,1,0,1,0,1,1,1,0,0,0,0,1],
    [1,1,1,1,0,1,0,0,0,0,0,0,0,1,0,1,1,1,1],
    [1,1,1,1,0,1,0,1,3,3,3,1,0,1,0,1,1,1,1],
    [0,0,0,0,0,0,0,1,3,3,3,1,0,0,0,0,0,0,0],
    [1,1,1,1,0,1,0,1,3,3,3,1,0,1,0,1,1,1,1],
    [1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1],
    [1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
    [1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,1],
    [1,0,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,0,1],
    [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1],
    [1,2,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],  
    [1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1]   
]


def recompute_total_pellets():
    return sum(row.count(0) + row.count(2) for row in maze)

total_pellets = recompute_total_pellets()
pellets_eaten = 0


# Utility: world <-> grid
def grid_to_world(x, y):
    return (
        x * CELL_SIZE - (MAZE_WIDTH * CELL_SIZE) / 2 + CELL_SIZE/2,
        y * CELL_SIZE - (MAZE_HEIGHT * CELL_SIZE) / 2 + CELL_SIZE/2
    )

def now_ms():
    return glutGet(GLUT_ELAPSED_TIME)

def power_active():
    return now_ms() < power_mode_until

def iframe_active():
    return now_ms() < iframe_until

def drawWireCube(size):
    hs = size / 2.0  
    vertices = [
        [-hs, -hs, -hs],
        [ hs, -hs, -hs],
        [ hs,  hs, -hs],
        [-hs,  hs, -hs],
        [-hs, -hs,  hs],
        [ hs, -hs,  hs],
        [ hs,  hs,  hs],
        [-hs,  hs,  hs]
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,0),  # back face
        (4,5),(5,6),(6,7),(7,4),  # front face
        (0,4),(1,5),(2,6),(3,7)   # connecting edges
    ]
    glBegin(GL_LINES)
    for e in edges:
        for v in e:
            glVertex3fv(vertices[v])
    glEnd()

def draw_text(x, y, text, font=GLUT_BITMAP_HELVETICA_18):
    glColor3f(1, 1, 1)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, 1000, 0, 800)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glRasterPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(font, ord(ch))

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_sphere(radius, slices=20, stacks=20):
    quadric = gluNewQuadric()
    gluSphere(quadric, radius, slices, stacks)
    

def draw_cylinder(base_radius, top_radius, height, slices=20):
    quadric = gluNewQuadric()
    gluCylinder(quadric, base_radius, top_radius, height, slices, slices)
    


def draw_pacman():
    global pacman_mouth, pacman_mouth_dir

    glPushMatrix()
    x, y = grid_to_world(pacman_pos[0], pacman_pos[1])
    z = PACMAN_SIZE
    glTranslatef(x, y, z)
    glRotatef(pacman_angle, 0, 0, 1)

    
    glColor3f(1, 1, 0)

    
    mouth_angle = abs(pacman_mouth) * 30
    if mouth_angle < 5:
        draw_sphere(PACMAN_SIZE)
    else:
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0, 0)
        for angle in range(int(mouth_angle), 360 - int(mouth_angle) + 1, 5):
            rad = math.radians(angle)
            glVertex3f(PACMAN_SIZE * math.cos(rad), PACMAN_SIZE * math.sin(rad), 0)
        glEnd()
        for i in range(0, 10):
            glBegin(GL_TRIANGLE_STRIP)
            for angle in range(int(mouth_angle), 360 - int(mouth_angle) + 1, 10):
                rad = math.radians(angle)
                z1 = PACMAN_SIZE * math.sin(math.radians(i * 9))
                z2 = PACMAN_SIZE * math.sin(math.radians((i + 1) * 9))
                r1 = PACMAN_SIZE * math.cos(math.radians(i * 9))
                r2 = PACMAN_SIZE * math.cos(math.radians((i + 1) * 9))
                glVertex3f(r1 * math.cos(rad), r1 * math.sin(rad), z1)
                glVertex3f(r2 * math.cos(rad), r2 * math.sin(rad), z2)
            glEnd()

    glPopMatrix()

def draw_ghost(ghost):
    if ghost.eaten and ghost.respawn_timer > 0:
        return

    glPushMatrix()
    x, y = grid_to_world(ghost.x, ghost.y)
    glTranslatef(x, y, 0)

    if power_active() and not ghost.eaten and not ghost.respawned_immune:
        glColor3f(0, 0, 1)
    else:
        glColor3f(*ghost.color)

    glPushMatrix()
    glRotatef(-90, 1, 0, 0)
    draw_cylinder(GHOST_SIZE, GHOST_SIZE * 0.8, GHOST_SIZE * 2, 20)
    glPopMatrix()

    glTranslatef(0, 0, GHOST_SIZE * 2)
    draw_sphere(GHOST_SIZE)

    glColor3f(1, 1, 1)
    glPushMatrix(); glTranslatef(GHOST_SIZE * 0.3, GHOST_SIZE * 0.3, 0); draw_sphere(3); glPopMatrix()
    glPushMatrix(); glTranslatef(-GHOST_SIZE * 0.3, GHOST_SIZE * 0.3, 0); draw_sphere(3); glPopMatrix()

    glColor3f(0, 0, 0)
    glPushMatrix(); glTranslatef(GHOST_SIZE * 0.3, GHOST_SIZE * 0.4, 2); draw_sphere(1.5); glPopMatrix()
    glPushMatrix(); glTranslatef(-GHOST_SIZE * 0.3, GHOST_SIZE * 0.4, 2); draw_sphere(1.5); glPopMatrix()

    glPopMatrix()

# Maze & pellets
def draw_maze():
    for y in range(MAZE_HEIGHT):
        for x in range(MAZE_WIDTH):
            world_x = x * CELL_SIZE - (MAZE_WIDTH * CELL_SIZE) / 2
            world_y = y * CELL_SIZE - (MAZE_HEIGHT * CELL_SIZE) / 2
            v = maze[y][x]

            if v == 1:  # Wall
                glPushMatrix()
                glTranslatef(world_x + CELL_SIZE/2, world_y + CELL_SIZE/2, WALL_HEIGHT/2)
                glColor3f(0.1, 0.1, 0.9)
                glScalef(CELL_SIZE, CELL_SIZE, WALL_HEIGHT)
                glutSolidCube(1)
                glColor3f(0.2, 0.2, 1.0)
                drawWireCube(1.01)
                glPopMatrix()
            elif v == 0:  # Pellet
                glPushMatrix()
                glTranslatef(world_x + CELL_SIZE/2, world_y + CELL_SIZE/2, PELLET_SIZE + 5)
                glRotatef(glutGet(GLUT_ELAPSED_TIME) * 0.1, 0, 0, 1)
                glColor3f(1, 1, 0.7)
                draw_sphere(PELLET_SIZE)
                glPopMatrix()
            elif v == 2:  # Power pellet
                glPushMatrix()
                glTranslatef(world_x + CELL_SIZE/2, world_y + CELL_SIZE/2, POWER_PELLET_SIZE + 5)
                pulse = 1.0 + 0.2 * math.sin(glutGet(GLUT_ELAPSED_TIME) * 0.005)
                glScalef(pulse, pulse, pulse)
                glColor3f(1, 0.7, 0)
                draw_sphere(POWER_PELLET_SIZE)
                glPopMatrix()
            elif v == 4:  # Tunnel tiles
                glPushMatrix()
                glTranslatef(world_x + CELL_SIZE/2, world_y + CELL_SIZE/2, 0)
                glColor3f(0.05, 0.05, 0.05)
                glBegin(GL_QUADS)
                glVertex3f(-CELL_SIZE/2, -CELL_SIZE/2, 0)
                glVertex3f(CELL_SIZE/2, -CELL_SIZE/2, 0)
                glVertex3f(CELL_SIZE/2, CELL_SIZE/2, 0)
                glVertex3f(-CELL_SIZE/2, CELL_SIZE/2, 0)
                glEnd()
                glPopMatrix()

    # Floor
    glColor3f(0.05, 0.05, 0.05)
    glBegin(GL_QUADS)
    half_w = (MAZE_WIDTH * CELL_SIZE) / 2
    half_h = (MAZE_HEIGHT * CELL_SIZE) / 2
    glVertex3f(-half_w, -half_h, 0)
    glVertex3f(half_w, -half_h, 0)
    glVertex3f(half_w, half_h, 0)
    glVertex3f(-half_w, half_h, 0)
    glEnd()


# BFS with optional randomized neighbor order and flee option

def bfs_pathfinding(start, target, flee=False, rng=None, shuffle_dirs=False):
    if start == target:
        return [0, 0]

    queue = deque([start])
    visited = {tuple(start): None}

    while queue:
        current = queue.popleft()

        dirs = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        if shuffle_dirs and rng is not None:
            rng.shuffle(dirs)

        for dx, dy in dirs:
            next_pos = [current[0] + dx, current[1] + dy]

            
            if next_pos[0] < 0:
                next_pos[0] = MAZE_WIDTH - 1
            elif next_pos[0] >= MAZE_WIDTH:
                next_pos[0] = 0

            
            center_x = MAZE_WIDTH // 2
            if next_pos[0] == center_x:
                if next_pos[1] < 0:
                    next_pos[1] = MAZE_HEIGHT - 1
                elif next_pos[1] >= MAZE_HEIGHT:
                    next_pos[1] = 0

            if (0 <= next_pos[1] < MAZE_HEIGHT and
                maze[next_pos[1]][next_pos[0]] != 1 and
                tuple(next_pos) not in visited):

                visited[tuple(next_pos)] = current
                queue.append(next_pos)

                if next_pos == target:
            
                    path = []
                    node = tuple(target)
                    while visited[node] is not None:
                        prev = visited[node]
                        path.append([node[0] - prev[0], node[1] - prev[1]])
                        node = tuple(prev)

                    if path:
                        if flee:
                            return [-path[0][0], -path[0][1]]  
                        else:
                            return path[-1]  

    return [0, 0]


def compute_distance_map(target_x, target_y, block_tunnels=True):
    
    from collections import deque as _deque
    q = _deque()
    start = (target_x, target_y)
    q.append(start)
    dist = {start: 0}
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((0,1),(0,-1),(-1,0),(1,0)):
            nx, ny = cx+dx, cy+dy
            
            if not (0 <= nx < MAZE_WIDTH and 0 <= ny < MAZE_HEIGHT):
                continue
            v = maze[ny][nx]
            if v == 1:  
                continue
            if block_tunnels and v == 4:  
                continue
            if (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(cx, cy)] + 1
                q.append((nx, ny))
    return dist


def update_ghosts():
    global power_mode_until
    now = now_ms()
    dmap = compute_distance_map(pacman_pos[0], pacman_pos[1], block_tunnels=True)

    for ghost in ghosts:
        if ghost.respawned_immune and now >= ghost.immune_until:
            ghost.respawned_immune = False
        if ghost.eaten:
            ghost.respawn_timer -= 1
            if ghost.respawn_timer <= 0:
                ghost.respawn()
            continue

        flee_now = (now_ms() < power_mode_until) and not ghost.respawned_immune

        candidates = []
        for dx, dy in ((0,1),(0,-1),(-1,0),(1,0)):
            nx, ny = ghost.x + dx, ghost.y + dy
            if not (0 <= nx < MAZE_WIDTH and 0 <= ny < MAZE_HEIGHT):
                continue
            v = maze[ny][nx]
            if v == 1 or v == 4:
                continue
            candidates.append((nx, ny, dx, dy))

    
        if ghost.recent:
            prev = ghost.recent[-1]
            non_back = [(nx,ny,dx,dy) for (nx,ny,dx,dy) in candidates if (nx,ny) != prev]
            if non_back:
                candidates = non_back

        def score(pos):
            return dmap.get((pos[0], pos[1]), -9999)

        if flee_now:
            best = None; best_val = -1e9
            for nx,ny,dx,dy in candidates:
                s = score((nx,ny)) + ghost.rng.random()*0.2
                if s > best_val:
                    best_val = s; best = (nx,ny,dx,dy)
        else:
            best = None; best_val = 1e9
            for nx,ny,dx,dy in candidates:
                s = score((nx,ny)) + ghost.rng.random()*0.2
                if s < best_val:
                    best_val = s; best = (nx,ny,dx,dy)

        if best is None and candidates:
            best = candidates[int(ghost.rng.random()*len(candidates))]

        if best:
            nx, ny, dx, dy = best
            ghost.recent.append((ghost.x, ghost.y))
            ghost.x, ghost.y = nx, ny

def check_collisions():
    global game_score, pellets_eaten, power_mode_until, lives, game_over, game_won, iframe_until

    
    px, py = pacman_pos
    x, y = int(px), int(py)
    if not (0 <= x < MAZE_WIDTH and 0 <= y < MAZE_HEIGHT):
        return

    # Pellet
    if maze[y][x] == 0:
        maze[y][x] = 3
        game_score += 10
        pellets_eaten += 1
    elif maze[y][x] == 2:
        maze[y][x] = 3
        game_score += 50
        pellets_eaten += 1
        power_mode_until = now_ms() + POWER_MODE_MS

    
    for ghost in ghosts:
        if not ghost.eaten and ghost.x == x and ghost.y == y:
            
            if power_active() and not getattr(ghost, "respawned_immune", False):
                ghost.eaten = True
                ghost.respawn_timer = int(1000 / GHOST_TICK_MS * 1.5)
                game_score += 200
                break
            elif not iframe_active():
                lives -= 1
                if lives < 0:
                    lives = 0
                # Start i-frames; 
                iframe_until = now_ms() + IFRAME_MS
                pacman_dir[:] = [0, 0]
                if lives <= 0:
                    game_over = True
                break


    if pellets_eaten >= total_pellets and total_pellets > 0:
        game_won = True

def reset_positions():
    global pacman_pos, pacman_dir, pacman_angle
    pacman_pos = [1, 1]
    pacman_dir = [0, 0]
    pacman_angle = 0
    for ghost in ghosts:
        ghost.respawn()
        ghost.respawned_immune = False
        ghost.immune_until = 0

def randomize_power_pellets(count=4):
    for y in range(MAZE_HEIGHT):
        for x in range(MAZE_WIDTH):
            if maze[y][x] == 2:
                maze[y][x] = 0

    walls = {(x, y) for y in range(MAZE_HEIGHT) for x in range(MAZE_WIDTH) if maze[y][x] == 1}
    tunnels = {(x, y) for y in range(MAZE_HEIGHT) for x in range(MAZE_WIDTH) if maze[y][x] == 4}
    def in_ghost_house(cx, cy):
        return (8 <= cx <= 10) and (8 <= cy <= 10)
    candidates = [(x, y) for y in range(MAZE_HEIGHT) for x in range(MAZE_WIDTH)
                  if maze[y][x] == 0 and (x, y) not in walls and (x, y) not in tunnels and not in_ghost_house(x, y)]
    random.shuffle(candidates)
    for i, (cx, cy) in enumerate(candidates[:count]):
        maze[cy][cx] = 2

def rebuild_maze_pellets():
    global total_pellets, pellets_eaten, last_ghost_step_ms
    for y in range(MAZE_HEIGHT):
        for x in range(MAZE_WIDTH):
            if maze[y][x] == 3:
                  if maze[y][x] != 1 and maze[y][x] != 4:
                    if not (8 <= x <= 10 and 8 <= y <= 10):
                        maze[y][x] = 0
    randomize_power_pellets(count=4)
    last_ghost_step_ms = now_ms()
    total_pellets = recompute_total_pellets()
    pellets_eaten = 0

def restart_game():
    global game_score, lives, power_mode_until, iframe_until, game_over, game_won, camera_mode 
    camera_mode = "2D"
    game_score = 0
    lives = 3
    power_mode_until = 0
    iframe_until = 0
    game_over = False
    game_won = False
    rebuild_maze_pellets()
    reset_positions()
    for g in ghosts:
        g.respawned_immune = False
        g.immune_until = 0



def animate():
    global pacman_mouth, pacman_mouth_dir, camera_mode, last_ghost_step_ms

    if game_paused or game_over or game_won:
        # Freeze ghost 
        last_ghost_step_ms = now_ms()
        glutPostRedisplay()
        return

    
    now = now_ms()
    while now - last_ghost_step_ms >= GHOST_TICK_MS:
        update_ghosts()
        last_ghost_step_ms += GHOST_TICK_MS

    # mouth animation
    pacman_mouth += pacman_mouth_dir * 0.1
    if pacman_mouth >= 1 or pacman_mouth <= 0:
        pacman_mouth_dir *= -1

    check_collisions()
    glutPostRedisplay()

def keyboardListener(key, x, y):
    global pacman_dir, pacman_angle, game_paused, camera_mode, pacman_last_move_ms

    if game_over or game_won:
        if key == b'r':
            restart_game()
        return

    if key == b' ':
        game_paused = not game_paused
        return

    if key == b'c':
        camera_mode = "2D" if camera_mode == "3D" else "3D"
        return

    if key == b'r':
        restart_game()
        return

    if game_paused:
        return

  
    now = now_ms()
    if now - pacman_last_move_ms < PACMAN_MOVE_COOLDOWN_MS:
        return

    
    if camera_mode == "2D":
    
        if key == b'w':
            nx, ny = pacman_pos[0], pacman_pos[1] + 1
            pacman_angle = 90
        elif key == b's':
            nx, ny = pacman_pos[0], pacman_pos[1] - 1
            pacman_angle = 270
        elif key == b'a':
            nx, ny = pacman_pos[0] - 1, pacman_pos[1]
            pacman_angle = 180
        elif key == b'd':
            nx, ny = pacman_pos[0] + 1, pacman_pos[1]
            pacman_angle = 0
        else:
            return
    else:
       
        if key == b'a':
            
            pacman_angle = (pacman_angle + 90) % 360
            pacman_last_move_ms = now
            return
        elif key == b'd':
            
            pacman_angle = (pacman_angle - 90) % 360
            pacman_last_move_ms = now
            return
        elif key == b'w':
        
            rad = math.radians(pacman_angle)
            dx = round(math.cos(rad))
            dy = round(math.sin(rad))
            nx, ny = pacman_pos[0] + dx, pacman_pos[1] + dy
        elif key == b's':
            rad = math.radians(pacman_angle)
            dx = round(math.cos(rad))
            dy = round(math.sin(rad))
            nx, ny = pacman_pos[0] - dx, pacman_pos[1] - dy
        else:
            return

    
    if nx < 0:
        nx = MAZE_WIDTH - 1
    elif nx >= MAZE_WIDTH:
        nx = 0
   
    center_x = MAZE_WIDTH // 2
    if nx == center_x:
        if ny < 0:
            ny = MAZE_HEIGHT - 1
        elif ny >= MAZE_HEIGHT:
            ny = 0

    if 0 <= ny < MAZE_HEIGHT and maze[ny][nx] != 1:
        pacman_pos[0] = nx
        pacman_pos[1] = ny
        pacman_last_move_ms = now


def setupCamera():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    half_w = (MAZE_WIDTH * CELL_SIZE) / 2
    half_h = (MAZE_HEIGHT * CELL_SIZE) / 2

    if camera_mode == "2D":
        glOrtho(-half_w, half_w, -half_h, half_h, -1000, 1000)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 600, 0, 0, 0, 0, 1, 0)
    else:
        gluPerspective(fovY, 1.25, 0.1, 2000)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        pac_x, pac_y = grid_to_world(pacman_pos[0], pacman_pos[1])
        cam_distance = 110  
        cam_height = 88  
        
        cam_x = pac_x - cam_distance * math.cos(math.radians(pacman_angle ))
        cam_y = pac_y - cam_distance * math.sin(math.radians(pacman_angle ))
        cam_z = cam_height
        look_ahead = 60
        look_x = pac_x + look_ahead * math.cos(math.radians(pacman_angle))
        look_y = pac_y + look_ahead * math.sin(math.radians(pacman_angle))
        gluLookAt(cam_x, cam_y, cam_z, look_x, look_y, 20, 0, 0, 1)


def idle():
    animate()

def showScreen():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glViewport(0, 0, 1000, 800)

    setupCamera()
    glEnable(GL_DEPTH_TEST)

    draw_maze()
    draw_pacman()
    for ghost in ghosts:
        draw_ghost(ghost)

    # HUD
    draw_text(10, 770, f"Score: {game_score}")
    draw_text(10, 740, f"Lives: {max(0, lives)}")
    draw_text(10, 710, f"Mode: {camera_mode} (C)")
    if power_active():
        secs = max(0, (power_mode_until - now_ms()) // 1000)
        draw_text(10, 680, f"POWER MODE: {secs}s")

    if iframe_active():
        if not game_over:
            secs = max(0, (iframe_until - now_ms()) // 1000)
            draw_text(10, 650, f"I-FRAMES: {secs}s")

    if game_paused:
        draw_text(430, 400, "GAME PAUSED!", GLUT_BITMAP_TIMES_ROMAN_24)
        draw_text(400, 370, "Press SPACE to continue!!!")
    if game_over:
        draw_text(450, 400, "GAME OVER", GLUT_BITMAP_TIMES_ROMAN_24)
        draw_text(390, 370, "Lives reached 0. Press R to restart.")
    elif game_won:
        draw_text(465, 400, "YOU WIN!", GLUT_BITMAP_TIMES_ROMAN_24)
        draw_text(390, 370, "All pellets eaten. Press R to restart.")

    draw_text(10, 30, "Controls: WASD-Move, SPACE-Pause, R-Restart, C-Change View")
    glutSwapBuffers()

def initialize_game():
    global camera_mode, power_mode_until, iframe_until, game_over, game_won, game_paused, last_ghost_step_ms
    camera_mode = "2D"
    power_mode_until = 0
    iframe_until = 0
    game_over = False
    game_won = False
    game_paused = False
    randomize_power_pellets(count=4)
    last_ghost_step_ms = now_ms()

def main():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1000, 800)
    glutInitWindowPosition(100, 100)
    glutCreateWindow(b"3D_Pac_Man")

    glEnable(GL_DEPTH_TEST)
    glClearColor(0, 0, 0, 1)

    initialize_game()

    glutDisplayFunc(showScreen)
    glutKeyboardFunc(keyboardListener)

    glutIdleFunc(idle)

    glutMainLoop()

if __name__ == "__main__":
    main()
