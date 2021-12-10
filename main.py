import dataclasses
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


class Game:
    def __init__(self) -> None:
        self._grid = Grid(size=[4, 4])

    def __str__(self):
        return self._grid.__str__()

    def get_empty_spots(self) -> list:
        pass

    def fill_initial(self):
        pass


if __name__ == "__main__":
    game = Game()
    pass
