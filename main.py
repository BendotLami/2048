import copy
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

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        return self.grid_size == other.grid_size and np.array_equal(self.array, other.array)

    def fill_debug(self):
        self.array = np.arange(0, self.grid_size ** 2).reshape([self.grid_size, self.grid_size])

    def get_empty_spot(self) -> np.ndarray:
        return np.argwhere(self.array == 0)

    def apply_function_by_direction(self, direction, func):
        for i in range(self.grid_size):
            if direction == 'r':
                curr_row = self.array[i, :]
                self.array[i, :] = func(curr_row[::-1])[::-1]
            elif direction == 'l':
                curr_row = self.array[i, :]
                self.array[i, :] = func(curr_row)
            elif direction == 'd':
                curr_row = self.array[:, i]
                self.array[:, i] = func(curr_row[::-1])[::-1]
            elif direction == 'u':
                curr_row = self.array[:, i]
                self.array[:, i] = func(curr_row)

    def move(self, direction):
        def move_to_end(array):
            non_zero = array[np.where(array != 0)]
            padding = np.zeros([self.grid_size - len(non_zero)], dtype=int)
            return np.concatenate([non_zero, padding])

        self.apply_function_by_direction(direction, move_to_end)


class Game:
    _initial_number_of_cells = 2

    def __init__(self) -> None:
        self._grid = Grid(grid_size=4)
        self._score = 0

    def __str__(self):
        return "\n".join([str(self.get_score()), self._grid.__str__()])

    def get_score(self):
        return self._score

    def _add_score(self, merge_value):
        self._score += merge_value

    def get_empty_spots(self) -> np.ndarray:
        return self._grid.get_empty_spot()

    def _fill_random_cells(self, size):
        spots = self.get_empty_spots()
        np.random.shuffle(spots)
        empty_spots = spots[0: size]
        random_values = np.random.choice([2, 4], p=[0.75, 0.25], size=size)
        self._grid.array[empty_spots[:, 0], empty_spots[:, 1]] = random_values

    def fill_initial(self):
        self._fill_random_cells(self._initial_number_of_cells)

    def fill_random_cell(self):
        self._fill_random_cells(1)

    def _merge_relevant(self, direction):
        """
        To be called after move!
        :param direction: Move direction
        :return: None
        """
        def _merge_array(array):
            """
            default is right move
            """
            i = len(array) - 1  # last element
            while i > 0:  # iterate backward
                if array[i] == array[i - 1]:  # same number, merge
                    array[i] *= 2
                    array[i - 1] = 0
                    self._add_score(array[i])
                    i -= 1
                i -= 1
            return array

        self._grid.apply_function_by_direction(direction, _merge_array)

    def move(self, direction):
        grid_before = copy.deepcopy(self._grid)
        self._grid.move(direction)
        self._merge_relevant(direction)
        self._grid.move(direction)
        if grid_before != self._grid:
            game.fill_random_cell()


if __name__ == "__main__":
    game = Game()
    game.fill_initial()
    # game._grid.fill_debug()
    print(game)

    while True:
        direction = input("Next move:")
        if direction not in ['l', 'r', 'u', 'd']:
            continue
        game.move(direction)
        print(game)
