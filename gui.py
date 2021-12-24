from main import GameMechanic
import tkinter as tk
from tkinter import ttk


class GameGui:
    _left_keys = ["<h>", "<a>", "<Left>"]
    _right_keys = ["<l>", "<d>", "<Right>"]
    _up_keys = ["<w>", "<k>", "<Up>"]
    _down_keys = ["<s>", "<j>", "<Down>"]

    def init_window(self):
        window = tk.Tk()
        for i in range(self.grid_size):
            window.rowconfigure(i, minsize=50)
            window.columnconfigure(i, minsize=50)
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

    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.game = GameMechanic(grid_size=self.grid_size)

        self._window = self.init_window()
        self._labels = self.init_labels()
        self._bind_keys()
        self._render_display()

    def _render_display(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_value = self.game.get_cell_value(i, j)
                self._labels[i][j]["text"] = f"{cell_value if cell_value else ''}"
                self._labels[i][j]["bg"] = "gray" if cell_value == 0 else "yellow"

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
    ggui = GameGui(2)
    ggui.mainloop()
    exit(0)
