import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, OptionMenu, StringVar, Button

class GridWorld:
    def __init__(self, size=101, num_envs=50, seed=None):
        self.size = size
        self.num_envs = num_envs
        self.seed = seed
        self.environments = []
        self.current_index = 0

    def generate_environments(self):
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
        if not self.environments:
            print("Environments not generated yet. Call generate_environment() first.")
            return

        grid = self.environments[index]
        plt.imshow(grid, cmap='binary', origin='lower')
        plt.title(f"Gridworld Environment {index + 1}")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()

    def on_next(self):
        self.current_index = (self.current_index + 1) % self.num_envs
        self.visualize_environment(self.current_index)

    def on_previous(self):
        self.current_index = (self.current_index - 1) % self.num_envs
        self.visualize_environment(self.current_index)

    def save_environments(self, filename="environments.npy"):
        np.save(filename, self.environments)

    def load_environments(self, filename="environments.npy"):
        self.environments = np.load(filename)


def main():
    # Usage
    grid_world = GridWorld(seed=42)
    grid_world.generate_environments()

    # Create a Tkinter window for the dropdown menu
    root = Tk()
    root.title("Grid Selection")

    # Dropdown menu
    options = [f"Environment {i + 1}" for i in range(grid_world.num_envs)]
    selected_option = StringVar(root)
    selected_option.set(options[0])
    dropdown = OptionMenu(root, selected_option, *options)
    dropdown.pack()

    # Buttons
    next_button = Button(root, text="Next", command=grid_world.on_next)
    next_button.pack()

    previous_button = Button(root, text="Previous", command=grid_world.on_previous)
    previous_button.pack()

    visualize_button = Button(root, text="Visualize", command=lambda: grid_world.visualize_environment(options.index(selected_option.get())))
    visualize_button.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
