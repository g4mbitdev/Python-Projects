import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import tracemalloc

# Parameters for the grid environment
GRID_SIZE = 10  # 10x10 grid
OBSTACLE_POSITIONS = [(2, 2), (2, 3), (2, 4), (5, 5), (6, 5), (7, 5), (8, 9), (3, 0)]
START_POS = (0, 0)  # Starting position of the robot
GOAL_POS = (9, 9)  # Goal position of the robot
goal_reached = False  # Global variable to track if the goal is reached

X = [0, 1, 0, -1]
Y = [-1, 0, 1, 0]

# Node class representing a grid cell
class Node:
    def __init__(self, obs, start, goal, x, y):
        self.obs = obs
        self.start = start
        self.goal = goal
        self.x = x
        self.y = y

# Environment class
class Environment:
    def __init__(self):
        self.startpos = START_POS
        self.goalpos = GOAL_POS
        self.grid = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                obs = (i, j) in OBSTACLE_POSITIONS
                start = (i, j) == START_POS
                goal = (i, j) == GOAL_POS
                self.grid.append(Node(obs, start, goal, i, j))
    
    def possible_moves(self, x, y):
        possible_moves = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in OBSTACLE_POSITIONS:
                possible_moves.append((nx, ny))
        return possible_moves

    # BFS Graph Function
    def graph_bfs(self, vis, i, j, visited):
        global goal_reached
        graphTimer = time.time() # Start Timer
        tracemalloc.start()

        if goal_reached:
            return

        queue = [(i, j)]
        visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]
        visited[i][j] = True

        parent = {}

        while queue:
            x, y = queue.pop(0)
            # print(queue)
            vis.update_visualization(x, y)

            if (x, y) == GOAL_POS:
                print("Goal reached!")
                goal_reached = True

                elapsedTime = time.time() - graphTimer

                path = []
                while(x, y) in parent:
                    path.append((x, y))
                    x, y = parent[(x, y)]

                path.append((i, j))
                path.reverse()

                # Calculate memory usage
                current, peak = tracemalloc.get_traced_memory()
                peakMemory = peak/1024
                currentMemory = current/1024
                tracemalloc.stop()

                # Print relevant info to terminal
                print("Here's your graph BFS optimal path: ", path)
                print(f"Path Length: {len(path)}")
                print(f"Elapsed Time: {elapsedTime:.4f} seconds.")
                print(f"Memory Usage in kilobytes (Current, Peak): {currentMemory:.1f}, {peakMemory:.1f}")

                for px, py in path:
                    vis.optimal_path(px, py)
                return
            
            for nx, ny in self.possible_moves(x, y):
                if not visited[nx][ny]:
                    visited[nx][ny] = True
                    parent[(nx, ny)] = (x, y)
                    queue.append((nx,ny))

    # BFS Tree Function
    def tree_bfs(self, vis, i, j):
        global goal_reached
        treeTimer = time.time() # Start Timer
        tracemalloc.start()

        if goal_reached:
            return

        queue = [(i, j)]
        parent = {}

        while queue:
            x, y = queue.pop(0)
            #print(queue)
            vis.update_visualization(x, y)

            if (x, y) == GOAL_POS:
                print("Goal reached!")
                goal_reached = True
                elapsedTime = time.time() - treeTimer

                path = []
                position = (x,y)

                while position in parent:
                    path.append((x, y))
                    position = parent[position]

                if position == (i, j):  
                    path.append((i, j))
                else:
                    print("There was an error reconstructing the path")
                    return
                path.reverse()

                # Calculate memory usage
                current, peak = tracemalloc.get_traced_memory()
                peakMemory = peak/1024
                currentMemory = current/1024
                tracemalloc.stop()

                # Print relevant info to terminal
                print("Here's your tree BFS optimal path: ", path)
                print(f"Path Length: {len(path)}")
                print(f"Elapsed Time: {elapsedTime:.4f} seconds.")
                print(f"Memory Usage in kilobytes (Current, Peak): {currentMemory:.1f}, {peakMemory:.1f}")
                print("Here's your tree BFS optimal path: ", path)

                for px, py in path:
                    vis.optimal_path(px, py)
                return
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and (nx, ny) not in OBSTACLE_POSITIONS and (nx, ny) not in parent:
                    parent[(nx, ny)] = (x, y)
                    queue.append((nx, ny))


# Visualizer class
class  Visualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.grid(which='both')

    def draw_environment(self, env):
        # Clear the existing figure
        self.ax.clear()
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.grid(which='both')

        # Draw the grid
        for cell in env.grid:
            if cell.obs:  # Obstacle
                self.ax.add_patch(patches.Rectangle((cell.y, GRID_SIZE - 1 - cell.x), 1, 1, facecolor='black'))
            elif cell.start:  # Start
                self.ax.add_patch(patches.Rectangle((cell.y, GRID_SIZE - 1 - cell.x), 1, 1, facecolor='red', edgecolor='black'))
            elif cell.goal:  # Goal
                self.ax.add_patch(patches.Rectangle((cell.y, GRID_SIZE - 1 - cell.x), 1, 1, facecolor='green', edgecolor='black'))
            else:
                self.ax.add_patch(patches.Rectangle((cell.y, GRID_SIZE - 1 - cell.x), 1, 1, facecolor='yellow', edgecolor='gray'))

    # Update the grid dynamically
    def update_visualization(self, x, y):
        self.ax.add_patch(patches.Rectangle((y, GRID_SIZE - 1 - x), 1, 1, facecolor='blue', edgecolor='gray'))
        plt.draw()
        #plt.pause(0.1)

    # Trace optimal path for BFS
    def optimal_path(self, x, y):
        self.ax.add_patch(patches.Rectangle((y, GRID_SIZE - 1 - x), 1, 1, facecolor='magenta', edgecolor='gray'))
        plt.draw()
        #plt.pause(0.1)

    def reset(self):
        self.draw_environment(env)
        plt.draw()
        #plt.pause(0.2)

# Main execution
env = Environment()
vis = Visualizer()

# Initial visualization
vis.draw_environment(env)
plt.show(block=False)

# Run Graph BFS
goal_reached = False
env.graph_bfs(vis, START_POS[0], START_POS[1], [])
vis.reset()

# Run Tree BFS
goal_reached = False
env.tree_bfs(vis, START_POS[0], START_POS[1])

vis.reset()

# Show final result
plt.show()

