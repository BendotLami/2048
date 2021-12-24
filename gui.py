from main import Game
import tkinter as tk
from tkinter import ttk


class GameGui:

    def __init__(self):
        self.window = tk.Tk()
        self.game = Game()
        for i in range(4):
            self.window.rowconfigure(i, minsize=50)
            self.window.columnconfigure(i, minsize=50)
        self.labels = []
        for i in range(4):
            lst = []
            for j in range(4):
                label = tk.Label(text=f"{i},{j}", bg="yellow", fg="black")
                label.grid(row=i, column=j, sticky="nsew", padx=3, pady=3)
                lst.append(label)
            self.labels.append(lst)
        self._render_display()
        self.window.bind("<h>", lambda x: self._make_a_move("l"))
        self.window.bind("<j>", lambda x: self._make_a_move("d"))
        self.window.bind("<k>", lambda x: self._make_a_move("u"))
        self.window.bind("<l>", lambda x: self._make_a_move("r"))
        self.window.mainloop()

    def _render_display(self):
        for i in range(4):
            for j in range(4):
                cell_value = self.game.get_cell_value(i, j)
                self.labels[i][j]["text"] = f"{cell_value if cell_value else ''}"
                self.labels[i][j]["bg"] = "gray" if cell_value == 0 else "yellow"

    def _make_a_move(self, direction):
        print(f"Making a move! {direction}")
        self.game.move(direction=direction)
        self._render_display()
        print(self.game)


if __name__ == "__main__":
    ggui = GameGui()
    exit(0)
