import dataclasses
import random

import numpy as np


class Grid:
    def __init__(self, size=None) -> None:
        if not size:
            size = [4, 4]
        self._array = np.zeros(size, dtype=int)

    def __str__(self):
        rtn_str = ""
        max_len = len(str(np.max(self._array))) + 1
        for row in self._array:
            for cell in row:
                cell_len = len(str(cell))
                rtn_str += f'{cell}{" "*(max_len - cell_len)}'
            rtn_str += '\n'
        return rtn_str

    def get_empty_spot(self) -> np.ndarray:
        return np.argwhere(self._array == 0)


class Game:
    _initial_number_of_cells = 2

    def __init__(self) -> None:
        self._grid = Grid(size=[4, 4])

    def __str__(self):
        return self._grid.__str__()

    def get_empty_spots(self) -> np.ndarray:
        return self._grid.get_empty_spot()

    def fill_initial(self):
        spots = self.get_empty_spots()
        np.random.shuffle(spots)
        empty_spots = spots[0: self._initial_number_of_cells]
        random_values = np.random.choice([2, 4], p=[0.75, 0.25], size=2)
        self._grid._array[empty_spots[:, 0], empty_spots[:, 1]] = random_values


if __name__ == "__main__":
    game = Game()
    game.fill_initial()
    print(game)
