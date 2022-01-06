import copy
import numpy as np


class GameMechanic:
    _initial_number_of_cells = 2
    _spaces_between_cells = 1
    _directions = ['u', 'd', 'l', 'r']
    _spawn_numbers = [2, 4]
    _spawn_probabilities = [0.75, 0.25]

    def __init__(self, grid_size=None) -> None:
        if not grid_size:
            grid_size = 4
        self.grid_size = grid_size
        self.grid = np.zeros([self.grid_size, self.grid_size], dtype=int)
        self._score = 0
        self.fill_initial()

    def __eq__(self, other):
        if not isinstance(other, GameMechanic):
            return False
        return self.grid_size == other.grid_size and np.array_equal(self.grid,
                                                                    other.grid) and self.get_score() == other.get_score()

    def stringify_board(self):
        rtn_str = ""
        max_len = len(str(np.max(self.grid))) + self._spaces_between_cells
        for row in self.grid:
            for cell in row:
                cell_len = len(str(cell))
                rtn_str += f'{cell if cell != 0 else "_"}{" " * (max_len - cell_len)}'
            rtn_str += '\n' * self._spaces_between_cells
        return rtn_str

    def __str__(self):
        rtn_str = ""
        # score
        rtn_str += f"{str(self.get_score())}\n"
        # board
        rtn_str += self.stringify_board()
        return rtn_str

    # Score handling
    def _add_score(self, merge_value):
        self._score += merge_value

    # Board filling
    def _fill_debug(self):
        self.grid = np.arange(0, self.grid_size ** 2).reshape([self.grid_size, self.grid_size])

    def _fill_random_cells(self, size):
        spots = self.get_empty_spots()
        np.random.shuffle(spots)
        empty_spots = spots[0: size]
        random_values = np.random.choice(self._spawn_numbers, p=self._spawn_probabilities, size=size)
        self.grid[empty_spots[:, 0], empty_spots[:, 1]] = random_values

    def fill_initial(self):
        self._fill_random_cells(self._initial_number_of_cells)

    def fill_random_cell(self):
        self._fill_random_cells(1)

    def get_empty_spots(self) -> np.ndarray:
        return np.argwhere(self.grid == 0)

    # Move handling
    def apply_function_by_direction(self, direction, func):
        for i in range(self.grid_size):
            if direction == 'r':
                curr_row = self.grid[i, :]
                self.grid[i, :] = func(curr_row[::-1])[::-1]
            elif direction == 'l':
                curr_row = self.grid[i, :]
                self.grid[i, :] = func(curr_row)
            elif direction == 'd':
                curr_row = self.grid[:, i]
                self.grid[:, i] = func(curr_row[::-1])[::-1]
            elif direction == 'u':
                curr_row = self.grid[:, i]
                self.grid[:, i] = func(curr_row)

    def push_values_to_edge(self, direction):
        def push_to_end(array):
            non_zero = array[np.where(array != 0)]
            padding = np.zeros([self.grid_size - len(non_zero)], dtype=int)
            return np.concatenate([non_zero, padding])

        self.apply_function_by_direction(direction, push_to_end)

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
            i = 0  # last element
            while i < len(array) - 1:  # iterate backward
                if array[i] == array[i + 1]:  # same number, merge
                    array[i] *= 2
                    array[i + 1] = 0
                    self._add_score(array[i])
                    i += 1
                i += 1
            return array

        self.apply_function_by_direction(direction, _merge_array)

    def move(self, direction):
        grid_before = copy.deepcopy(self.grid)
        self.push_values_to_edge(direction)
        self._merge_relevant(direction)
        self.push_values_to_edge(direction)
        if not np.array_equal(grid_before, self.grid):
            self.fill_random_cell()
            return True
        return False

    # API for state
    def get_cell_value(self, x, y):
        return self.grid[x, y]

    def get_score(self):
        return self._score

    def get_grid_size(self):
        return self.grid_size

    def can_move_to_direction(self, direction):
        temp_game = copy.deepcopy(self)
        return temp_game.move(direction)

    def any_move(self):
        for move in self._directions:
            if self.can_move_to_direction(move):
                return True
        return False

    def is_game_over(self):
        return not self.any_move()


if __name__ == "__main__":
    game = GameMechanic()
    # game._grid.fill_debug()
    print(game)

    while True:
        user_input = input("Next move:")
        if user_input.strip().upper() == "EXIT" or game.is_game_over():
            exit(0)
        if user_input not in ['l', 'r', 'u', 'd']:
            continue
        game.move(user_input)
        print(game)
