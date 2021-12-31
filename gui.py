from main import GameMechanic
import tkinter as tk
from tkinter import ttk
from math import log2


def render_rgb(rgb):
    return "#%02x%02x%02x" % rgb


class GameGui:
    _left_keys = ["<h>", "<a>", "<Left>"]
    _right_keys = ["<l>", "<d>", "<Right>"]
    _up_keys = ["<w>", "<k>", "<Up>"]
    _down_keys = ["<s>", "<j>", "<Down>"]

    _gray = 160, 160, 160
    _bg_color_list = ["#eee4da", "#eee1c9", "#f3b27a", "#f69664", "#f77c5f", "#f75f3b", "#edd073", "#edcc62", "#edc950",
                      "#edc53f", "#edc22e"]

    def init_window(self):
        window = tk.Tk()
        for i in range(self.grid_size):
            window.rowconfigure(i, minsize=self.window_size)
            window.columnconfigure(i, minsize=self.window_size)
        return window

    def init_labels(self):
        labels = []
        for i in range(self.grid_size):
            lst = []
            for j in range(self.grid_size):
                label = tk.Label(text=f"", bg="yellow", fg="black")
                label.grid(row=i, column=j, sticky="nsew", padx=3, pady=3)
                lst.append(label)
            labels.append(lst)
        return labels

    def __init__(self, grid_size=4, window_size=100):
        self.grid_size = grid_size
        self.window_size = window_size
        self.game = GameMechanic(grid_size=self.grid_size)

        self._window = self.init_window()
        self._labels = self.init_labels()
        self._bind_keys()
        self._render_display()

    @staticmethod
    def _render_cell_text(value):
        return value if value else ''

    @staticmethod
    def _render_cell_color(value):  # for 128 >= value
        if value == 0:
            return render_rgb(GameGui._gray)
        log_value = int(log2(value)) - 1  # minus one for values [0,..]
        return GameGui._bg_color_list[log_value]

    @staticmethod
    def _render_cell_text_color(value):
        if value > 4:
            return '#f9f6f2'
        else:
            return '#000000'

    def _render_display(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_value = self.game.get_cell_value(i, j)
                self._labels[i][j]["text"] = self._render_cell_text(cell_value)
                self._labels[i][j]["bg"] = self._render_cell_color(cell_value)
                self._labels[i][j]["fg"] = self._render_cell_text_color(cell_value)

    def _make_a_move(self, direction):
        print(f"Making a move! {direction}")
        self.game.move(direction=direction)
        self._render_display()
        print(self.game)

    def move(self, direction):
        self._make_a_move(direction)
        if self.game.is_game_over():
            self._window.destroy()

    def _bind_keys(self):
        for key in self._left_keys:
            self._window.bind(key, lambda x: self.move("l"))
        for key in self._down_keys:
            self._window.bind(key, lambda x: self.move("d"))
        for key in self._up_keys:
            self._window.bind(key, lambda x: self.move("u"))
        for key in self._right_keys:
            self._window.bind(key, lambda x: self.move("r"))

    def mainloop(self):
        self._window.mainloop()


if __name__ == "__main__":
    ggui = GameGui()
    ggui.mainloop()
    exit(0)
