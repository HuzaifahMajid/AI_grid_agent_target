import numpy as np
import random

num_rows = 5
num_cols = 5
mazesNumber = 50

def isDeadEnd(y, x, visited):
    """
    Checks if the current cell is a dead end or has unvisited neighbors.

    Args:
        y (int): Current row index.
        x (int): Current column index.
        visited (set): Set of visited nodes.

    Returns:
        tuple: A tuple containing a boolean indicating if it's a dead end, and the coordinates of an unvisited neighbor.
    """
    for i in range(-1, 2):  # i in -1, 0, 1
        for j in range(-1, 2):  # j in -1, 0, 1
            if not (i == 0 and j == 0):  # as if i==0 and j==0 then we are in the same cell
                if 0 <= x + i < num_cols and 0 <= y + j < num_rows:
                    if (x + i, y + j) not in visited:
                        return False, y + j, x + i  # There's an unvisited neighbor
    return True, -1, -1  # There's no unvisited neighbor


def generateMazes():
    """
    Generates random mazes using a depth-first search algorithm.

    Returns:
        numpy.ndarray: A 3D numpy array representing the generated mazes.
    """
    maze = np.zeros((mazesNumber, num_rows, num_cols))

    for mazeInd in range(0, mazesNumber):
        print("Generate Maze: " + str(mazeInd + 1))
        visited = set()
        stack = []

        row_index = random.randint(0, num_rows - 1)
        col_index = random.randint(0, num_cols - 1)

        print("- Start -\n")
        print("Loc[" + str(row_index) + "],[" + str(col_index) + "] = 1")
        visited.add((row_index, col_index))
        maze[mazeInd, row_index, col_index] = 1

        print("\n\n-  DFS -\n")
        while len(visited) < num_cols * num_rows:
            crnt_row_index = row_index + random.randint(-1, 1)
            crnt_col_index = col_index + random.randint(-1, 1)

            i = 0
            isDead = False
            while (
                not (0 <= crnt_row_index < num_rows)
                or not (0 <= crnt_col_index < num_cols)
                or (crnt_row_index, crnt_col_index) in visited
            ):
                crnt_row_index = row_index + random.randint(-1, 1)
                crnt_col_index = col_index + random.randint(-1, 1)
                i += 1
                print("stuck" + str(i))
                if i == 8:
                    isDead = True
                    break

            if not isDead:
                visited.add((crnt_row_index, crnt_col_index))

            rand_num = random.uniform(0, 1)

            if rand_num < 0.3 and not isDead:
                maze[mazeInd, crnt_row_index, crnt_col_index] = 0
                print("Loc[" + str(crnt_row_index) + "],[" + str(crnt_col_index) + "] = 0")
                row_index = crnt_row_index
                col_index = crnt_col_index
            else:
                if not isDead:
                    maze[mazeInd, crnt_row_index, crnt_col_index] = 1
                    print("Loc[" + str(crnt_row_index) + "],[" + str(crnt_col_index) + "] = 1")
                    stack.append((crnt_row_index, crnt_col_index))
                    isDead, unvisitRow, unvisitCol = isDeadEnd(row_index, col_index, visited)

                if isDead:
                    while stack:
                        parent_row, parent_col = stack.pop()
                        isDead, unvisitRow, unvisitCol = isDeadEnd(parent_row, parent_col, visited)
                        if not isDead:
                            break

                    if stack:
                        visited.add((unvisitRow, unvisitCol))
                        row_index = unvisitRow
                        col_index = unvisitCol
                    else:
                        row_index = random.randint(0, num_rows - 1)
                        col_index = random.randint(0, num_cols - 1)
                        if len(visited) < num_cols * num_rows:
                            while (
                                not (0 <= row_index < num_rows)
                                or not (0 <= col_index < num_cols)
                                or (row_index, col_index) in visited
                            ):
                                row_index = random.randint(0, num_rows - 1)
                                col_index = random.randint(0, num_cols - 1)
            print("maze successfully generated")
    return maze


if __name__ == "__main__":
    mazes = generateMazes()

    for mazeInd in range(0, mazesNumber):
        np.savetxt("maze" + str(mazeInd) + ".txt", mazes[mazeInd].astype(int), fmt="%i", delimiter=":")
