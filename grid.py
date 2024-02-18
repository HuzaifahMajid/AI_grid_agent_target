import numpy as np


class GridWorld:
    def __init__(self, size=101, num_envs=50, seed=None):
        self.size = size
        self.num_envs = num_envs
        self.seed = seed
        self.environments = []

    def generate_environment(self):
        np.random.seed(self.seed)
        environments = []
        for _ in range(self.num_envs):
            # Initialize grid with all cells unvisited
            grid = np.zeros((self.size, self.size), dtype=bool)
            visited = np.zeros((self.size, self.size), dtype=bool)

            # Start from a random cell
            start_row, start_col = np.random.randint(0, self.size), np.random.randint(0, self.size)
            stack = [(start_row, start_col)]

            while stack:
                current_row, current_col = stack[-1]
                visited[current_row, current_col] = True

                # Get unvisited neighbors
                neighbors = [(r, c) for r, c in [(current_row + 1, current_col),
                                                 (current_row - 1, current_col),
                                                 (current_row, current_col + 1),
                                                 (current_row, current_col - 1)]
                             if 0 <= r < self.size and 0 <= c < self.size and not visited[r, c]]

                if neighbors:
                    next_row, next_col = neighbors[np.random.randint(0, len(neighbors))]
                    grid[next_row, next_col] = True

                    # With 30% probability mark as blocked
                    if np.random.random() < 0.3:
                        grid[next_row, next_col] = False

                    stack.append((next_row, next_col))
                else:
                    # Backtrack if there are no unvisited neighbors
                    stack.pop()

            environments.append(grid)

        self.environments = environments

    def visualize_environment(self, index):
        import matplotlib.pyplot as plt

        if not self.environments:
            print("Environments not generated yet. Call generate_environment() first.")
            return

        grid = self.environments[index]
        plt.imshow(grid, cmap='binary', origin='lower')
        plt.title(f"Gridworld Environment {index + 1}")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.colorbar(label="Blocked (0) / Unblocked (1)")
        plt.show()


# Usage
grid_world = GridWorld()
grid_world.generate_environment()
grid_world.visualize_environment(0)  # Visualize the first environment

