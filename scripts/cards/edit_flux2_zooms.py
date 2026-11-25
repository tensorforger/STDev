"""
Visual editor of the zoom regions of the Steppe-Flux.2 comparison cards (flux2_zooms.json, read by
make_flux2_comparisons.py). Box shapes follow the card layout: a box keeps the aspect ratio of its row, the row
weights change the shapes.

Mouse: drag a box to move it, drag a corner to resize it, wheel over a box to resize it.
Keys: Left / Right or 1-9 switch the image, Tab selects the next box, + / - change the row weight of the selected box,
a adds a box, Delete removes the selected box, Ctrl+S saves, r renders the full card preview.

Run: python scripts/cards/edit_flux2_zooms.py [--input-directory ../comfy_results_flux_2] [--zooms-file ...]
"""

import argparse
import tkinter
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox

import make_flux2_comparisons as comparisons
from make_flux2_comparisons import ComparisonCard, Zoom
from PIL import Image, ImageTk

BACKGROUND = "#0b0c0e"
PANEL = "#16181c"
TEXT = "#eae8e3"
MUTED = "#80868f"
ACCENT = "#e9a94b"
HANDLE_RADIUS = 7
PREVIEW_WIDTH = 380
PREVIEW_MAX_HEIGHT = 560
MINIMUM_ZOOM_WIDTH = 16.0
WEIGHT_STEP = 1.1
WHEEL_STEP = 1.05


@dataclass
class Drag:
    """mode: "move" or "resize"; anchor: fixed point in low resolution pixels (move: offset of the center)"""

    mode: str
    anchor: tuple[float, float]
    sign: tuple[float, float] = (1.0, 1.0)
    aspect: float = 1.0


class ZoomEditor:
    def __init__(
        self,
        root: tkinter.Tk,
        cards: list[ComparisonCard],
        input_directory: Path,
        zooms_file: Path,
    ) -> None:
        self.root = root
        self.cards = cards
        self.input_directory = input_directory
        self.zooms_file = zooms_file
        self.card_index = 0
        self.selected: int | None = 0
        self.drag: Drag | None = None
        self.dirty = False
        self.low_images: dict[str, Image.Image] = {}
        self.high_images: dict[str, Image.Image] = {}
        self.display_height = min(root.winfo_screenheight() - 140, 1150)
        self.scale = 1.0
        self.photo: ImageTk.PhotoImage | None = None
        self.preview_photos: list[ImageTk.PhotoImage] = []
        self.render_photo: ImageTk.PhotoImage | None = None
        self.build_interface()
        self.load_card()

    @property
    def card(self) -> ComparisonCard:
        return self.cards[self.card_index]

    def build_interface(self) -> None:
        self.root.configure(background=BACKGROUND)
        self.canvas = tkinter.Canvas(
            self.root, background=BACKGROUND, highlightthickness=0, cursor="crosshair"
        )
        self.canvas.pack(side="left", padx=10, pady=10)
        panel = tkinter.Frame(self.root, background=BACKGROUND)
        panel.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)

        image_row = tkinter.Frame(panel, background=BACKGROUND)
        image_row.pack(anchor="w")
        self.image_buttons = []
        for index, card_spec in enumerate(self.cards):
            button = tkinter.Button(
                image_row,
                text=card_spec.letter,
                width=3,
                command=lambda index=index: self.switch_card(index),
            )
            button.pack(side="left", padx=2)
            self.image_buttons.append(button)

        control_row = tkinter.Frame(panel, background=BACKGROUND)
        control_row.pack(anchor="w", pady=(8, 0))
        controls = (
            ("weight -", lambda: self.change_weight(1 / WEIGHT_STEP)),
            ("weight +", lambda: self.change_weight(WEIGHT_STEP)),
            ("add box", self.add_zoom),
            ("delete box", self.delete_zoom),
            ("render card", self.render_card),
            ("save", self.save),
        )
        for text, command in controls:
            tkinter.Button(control_row, text=text, command=command).pack(side="left", padx=2)

        self.info = tkinter.Label(
            panel,
            background=BACKGROUND,
            foreground=TEXT,
            justify="left",
            anchor="w",
            font=("monospace", 10),
        )
        self.info.pack(anchor="w", pady=(10, 6), fill="x")
        previews = tkinter.Frame(panel, background=BACKGROUND)
        previews.pack(anchor="w")
        self.preview_labels = []
        for caption in ("bicubic 4×", "steppe 4×"):
            column = tkinter.Frame(previews, background=BACKGROUND)
            column.pack(side="left", padx=(0, 8), anchor="n")
            tkinter.Label(column, text=caption, background=BACKGROUND, foreground=MUTED).pack(
                anchor="w"
            )
            label = tkinter.Label(column, background=PANEL)
            label.pack()
            self.preview_labels.append(label)
        tkinter.Label(
            panel,
            text=(
                "drag box: move · drag corner / wheel: resize · Tab: next box · +/-: row weight\n"
                "←/→ or 1-9: image · a: add · Delete: remove · r: render · Ctrl+S: save"
            ),
            background=BACKGROUND,
            foreground=MUTED,
            justify="left",
        ).pack(anchor="w", side="bottom")

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", lambda event: self.wheel_at(event.x, event.y, 1))
        self.canvas.bind("<Button-5>", lambda event: self.wheel_at(event.x, event.y, -1))
        self.root.bind("<Left>", lambda event: self.switch_card(self.card_index - 1))
        self.root.bind("<Right>", lambda event: self.switch_card(self.card_index + 1))
        for number in range(1, min(len(self.cards), 9) + 1):
            self.root.bind(str(number), lambda event, index=number - 1: self.switch_card(index))
        self.root.bind("<Tab>", self.select_next)
        for key in ("<plus>", "<equal>", "<KP_Add>"):
            self.root.bind(key, lambda event: self.change_weight(WEIGHT_STEP))
        for key in ("<minus>", "<KP_Subtract>"):
            self.root.bind(key, lambda event: self.change_weight(1 / WEIGHT_STEP))
        self.root.bind("a", lambda event: self.add_zoom())
        self.root.bind("<Delete>", lambda event: self.delete_zoom())
        self.root.bind("r", lambda event: self.render_card())
        self.root.bind("<Control-s>", lambda event: self.save())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def low_image(self) -> Image.Image:
        letter = self.card.letter
        if letter not in self.low_images:
            path = self.input_directory / f"{letter}_low.png"
            self.low_images[letter] = Image.open(path).convert("RGB")
        return self.low_images[letter]

    def high_image(self) -> Image.Image:
        letter = self.card.letter
        if letter not in self.high_images:
            self.root.configure(cursor="watch")
            self.root.update_idletasks()
            self.high_images = {
                letter: Image.open(self.input_directory / f"{letter}_high.png").convert("RGB")
            }
            self.root.configure(cursor="")
        return self.high_images[letter]

    def layout(self) -> comparisons.CardLayout:
        return comparisons.card_layout(
            self.card.zooms, self.low_image().size, self.high_image().size
        )

    def load_card(self) -> None:
        low = self.low_image()
        self.scale = self.display_height / low.height
        shown = low.resize(
            (round(low.width * self.scale), self.display_height), Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(shown)
        self.canvas.configure(width=shown.width, height=shown.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        for index, button in enumerate(self.image_buttons):
            button.configure(relief="sunken" if index == self.card_index else "raised")
        if self.selected is not None and self.selected >= len(self.card.zooms):
            self.selected = 0 if self.card.zooms else None
        self.redraw()
        self.update_preview()

    def switch_card(self, index: int) -> None:
        self.card_index = index % len(self.cards)
        self.selected = 0 if self.card.zooms else None
        self.load_card()

    def canvas_box(self, box: tuple[float, float, float, float]) -> tuple[float, ...]:
        return tuple(value * self.scale for value in box)

    def redraw(self) -> None:
        self.canvas.delete("box")
        layout = self.layout()
        for index, row in enumerate(layout.rows):
            left, top, right, bottom = self.canvas_box(row.box)
            is_selected = index == self.selected
            self.canvas.create_rectangle(
                left, top, right, bottom, outline=BACKGROUND, width=5 if is_selected else 4, tags="box"
            )
            self.canvas.create_rectangle(
                left, top, right, bottom, outline=ACCENT, width=3 if is_selected else 1.5, tags="box"
            )
            self.canvas.create_rectangle(
                left, top - 20, left + 20, top, fill=ACCENT, outline="", tags="box"
            )
            self.canvas.create_text(
                left + 10, top - 10, text=str(index + 1), fill=BACKGROUND, tags="box"
            )
            if is_selected:
                for corner_x, corner_y in ((left, top), (right, top), (left, bottom), (right, bottom)):
                    self.canvas.create_rectangle(
                        corner_x - HANDLE_RADIUS,
                        corner_y - HANDLE_RADIUS,
                        corner_x + HANDLE_RADIUS,
                        corner_y + HANDLE_RADIUS,
                        fill=TEXT,
                        outline=BACKGROUND,
                        tags="box",
                    )
        self.update_info(layout)
        self.root.title(
            f"Steppe-Flux.2 zooms · {self.card.letter} {self.card.name}{' *' if self.dirty else ''}"
        )

    def update_info(self, layout: comparisons.CardLayout) -> None:
        lines = []
        factor = self.high_image().width / self.low_image().width
        for index, (zoom, row) in enumerate(zip(self.card.zooms, layout.rows, strict=True)):
            left, top, right, bottom = row.box
            display_zoom = layout.tile_width / ((right - left) * factor)
            marker = "▶" if index == self.selected else " "
            lines.append(
                f"{marker} {index + 1}: center ({(left + right) / 2:6.1f}, {(top + bottom) / 2:6.1f})  "
                f"size {right - left:5.1f}×{bottom - top:5.1f}  weight {zoom.weight:.3f}  "
                f"display {display_zoom:.2f}× of 4× pixels"
            )
        self.info.configure(text="\n".join(lines) if lines else "no boxes: press a to add one")

    def update_preview(self) -> None:
        self.preview_photos = []
        if self.selected is None:
            for label in self.preview_labels:
                label.configure(image="")
            return
        layout = self.layout()
        row = layout.rows[self.selected]
        height = round(PREVIEW_WIDTH * row.height / layout.tile_width)
        width = PREVIEW_WIDTH
        if height > PREVIEW_MAX_HEIGHT:
            width, height = round(width * PREVIEW_MAX_HEIGHT / height), PREVIEW_MAX_HEIGHT
        pair = comparisons.zoom_pair(self.low_image(), self.high_image(), row.box, (width, height))
        for label, image in zip(self.preview_labels, pair, strict=True):
            photo = ImageTk.PhotoImage(image)
            self.preview_photos.append(photo)
            label.configure(image=photo)

    def mark_changed(self) -> None:
        self.dirty = True
        self.redraw()

    def hit_test(self, x: float, y: float) -> tuple[int, str, tuple[float, float]] | None:
        """returns: (zoom index, "corner" or "inside", corner signs) for the canvas point, selected box first"""
        layout = self.layout()
        order = list(range(len(layout.rows)))
        if self.selected is not None:
            order.remove(self.selected)
            order.insert(0, self.selected)
        for index in order:
            left, top, right, bottom = self.canvas_box(layout.rows[index].box)
            for corner_x, sign_x in ((left, -1.0), (right, 1.0)):
                for corner_y, sign_y in ((top, -1.0), (bottom, 1.0)):
                    if abs(x - corner_x) <= HANDLE_RADIUS + 3 and abs(y - corner_y) <= HANDLE_RADIUS + 3:
                        return index, "corner", (sign_x, sign_y)
        for index in order:
            left, top, right, bottom = self.canvas_box(layout.rows[index].box)
            if left <= x <= right and top <= y <= bottom:
                return index, "inside", (1.0, 1.0)
        return None

    def on_press(self, event: tkinter.Event) -> None:
        hit = self.hit_test(event.x, event.y)
        if hit is None:
            self.drag = None
            return
        index, part, sign = hit
        self.selected = index
        layout = self.layout()
        left, top, right, bottom = layout.rows[index].box
        mouse = (event.x / self.scale, event.y / self.scale)
        if part == "corner":
            anchor_x = left if sign[0] > 0 else right
            anchor_y = top if sign[1] > 0 else bottom
            aspect = layout.rows[index].height / layout.tile_width
            self.drag = Drag("resize", (anchor_x, anchor_y), sign, aspect)
        else:
            center = ((left + right) / 2, (top + bottom) / 2)
            self.drag = Drag("move", (center[0] - mouse[0], center[1] - mouse[1]))
        self.redraw()

    def on_motion(self, event: tkinter.Event) -> None:
        if self.drag is None or self.selected is None:
            return
        zoom = self.card.zooms[self.selected]
        mouse_x, mouse_y = event.x / self.scale, event.y / self.scale
        if self.drag.mode == "move":
            zoom.center = (mouse_x + self.drag.anchor[0], mouse_y + self.drag.anchor[1])
        else:
            anchor_x, anchor_y = self.drag.anchor
            width = max(
                abs(mouse_x - anchor_x), abs(mouse_y - anchor_y) / self.drag.aspect, MINIMUM_ZOOM_WIDTH
            )
            zoom.width = width
            zoom.center = (
                anchor_x + self.drag.sign[0] * width / 2,
                anchor_y + self.drag.sign[1] * width * self.drag.aspect / 2,
            )
        self.clamp_centers()
        self.mark_changed()

    def on_release(self, event: tkinter.Event) -> None:
        self.drag = None
        self.update_preview()

    def on_wheel(self, event: tkinter.Event) -> None:
        self.wheel_at(event.x, event.y, 1 if event.delta > 0 else -1)

    def wheel_at(self, x: float, y: float, direction: int) -> None:
        hit = self.hit_test(x, y)
        if hit is None:
            return
        self.selected = hit[0]
        zoom = self.card.zooms[self.selected]
        zoom.width = max(zoom.width * WHEEL_STEP ** (-direction), MINIMUM_ZOOM_WIDTH)
        self.clamp_centers()
        self.mark_changed()
        self.update_preview()

    def clamp_centers(self) -> None:
        """stores the shown (shifted inside the image) box centers, so the file matches what is shown"""
        layout = self.layout()
        for zoom, row in zip(self.card.zooms, layout.rows, strict=True):
            left, top, right, bottom = row.box
            zoom.center = ((left + right) / 2, (top + bottom) / 2)

    def select_next(self, event: tkinter.Event | None = None) -> str:
        if self.card.zooms:
            self.selected = 0 if self.selected is None else (self.selected + 1) % len(self.card.zooms)
            self.redraw()
            self.update_preview()
        return "break"

    def change_weight(self, multiplier: float) -> None:
        if self.selected is None:
            return
        zoom = self.card.zooms[self.selected]
        zoom.weight = min(max(zoom.weight * multiplier, 0.1), 10.0)
        self.clamp_centers()
        self.mark_changed()
        self.update_preview()

    def add_zoom(self) -> None:
        low = self.low_image()
        self.card.zooms.append(Zoom((low.width / 2, low.height / 2), 150.0, 1.0))
        self.selected = len(self.card.zooms) - 1
        self.clamp_centers()
        self.mark_changed()
        self.update_preview()

    def delete_zoom(self) -> None:
        if self.selected is None or len(self.card.zooms) <= 1:
            return
        del self.card.zooms[self.selected]
        self.selected = min(self.selected, len(self.card.zooms) - 1)
        self.mark_changed()
        self.update_preview()

    def save(self) -> None:
        comparisons.save_cards(self.cards, self.zooms_file)
        self.dirty = False
        self.redraw()

    def render_card(self) -> None:
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        image = comparisons.build(self.card, self.low_image(), self.high_image())
        self.root.configure(cursor="")
        window = tkinter.Toplevel(self.root, background=BACKGROUND)
        window.title(f"card preview · {self.card.name} (not saved)")
        width = min(self.root.winfo_screenwidth() - 80, image.width)
        shown = image.resize(
            (width, round(image.height * width / image.width)), Image.Resampling.LANCZOS
        )
        self.render_photo = ImageTk.PhotoImage(shown)
        tkinter.Label(window, image=self.render_photo, background=BACKGROUND).pack()
        window.bind("<Escape>", lambda event: window.destroy())

    def on_close(self) -> None:
        if self.dirty:
            answer = messagebox.askyesnocancel("Unsaved changes", "Save the zooms before closing?")
            if answer is None:
                return
            if answer:
                self.save()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-directory", type=Path, default=comparisons.DEFAULT_INPUT_DIRECTORY)
    parser.add_argument("--zooms-file", type=Path, default=comparisons.ZOOMS_FILE)
    arguments = parser.parse_args()
    root = tkinter.Tk()
    ZoomEditor(root, comparisons.load_cards(arguments.zooms_file), arguments.input_directory, arguments.zooms_file)
    root.mainloop()


if __name__ == "__main__":
    main()
