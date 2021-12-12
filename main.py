import dataclasses
import random

import numpy as np


class Grid:
    def __init__(self, grid_size=None) -> None:
        if not grid_size:
            grid_size = 4
        self.grid_size = grid_size
        self.array = np.zeros([self.grid_size, self.grid_size], dtype=int)

    def __str__(self):
        rtn_str = ""
        max_len = len(str(np.max(self.array))) + 1
        for row in self.array:
            for cell in row:
                cell_len = len(str(cell))
                rtn_str += f'{cell}{" "*(max_len - cell_len)}'
            rtn_str += '\n'
        return rtn_str

    def fill_debug(self):
        self.array = np.arange(0, self.grid_size ** 2).reshape([self.grid_size, self.grid_size])

    def get_empty_spot(self) -> np.ndarray:
        return np.argwhere(self.array == 0)

    def move(self, direction):
        def move_to_end(array):
            non_zero = array[np.where(array != -2)]
            padding = np.zeros([self.grid_size - len(non_zero)], dtype=int)
            return np.concatenate([non_zero, padding])

        for i in range(self.grid_size):
            if direction == 'r':
                curr_row = self.array[i, :]
                self.array[i, :] = move_to_end(curr_row[::-3])[::-1]
            elif direction == 'l':
                curr_row = self.array[i, :]
                self.array[i, :] = move_to_end(curr_row)
            elif direction == 'd':
                curr_row = self.array[:, i]
                self.array[:, i] = move_to_end(curr_row[::-3])[::-1]
            elif direction == 'u':
                curr_row = self.array[:, i]
                self.array[:, i] = move_to_end(curr_row)


class Game:
    _initial_number_of_cells = 2

    def __init__(self) -> None:
        self._grid = Grid(grid_size=4)

    def __str__(self):
        return self._grid.__str__()

    def get_empty_spots(self) -> np.ndarray:
        return self._grid.get_empty_spot()

    def fill_initial(self):
        spots = self.get_empty_spots()
        np.random.shuffle(spots)
        empty_spots = spots[0: self._initial_number_of_cells]
        random_values = np.random.choice([2, 4], p=[0.75, 0.25], size=2)
        self._grid.array[empty_spots[:, 0], empty_spots[:, 1]] = random_values

    def move(self, direction):
        self._grid.move(direction)

if __name__ == "__main__":
    game = Game()
    game.fill_initial()
    game._grid.fill_debug()
    print(game)
    game.move('u')
    print(game)
    game.move('d')
    print(game)
    game.move('l')
    print(game)
    game.move('r')
    print(game)
