#!/usr/bin/env python3
"""
UNO Card Symbol Labeler
=======================

A fast keyboard-driven GUI for labeling UNO card symbol crops.

Usage:
    python label_uno.py /path/to/images
    python label_uno.py /path/to/images --labels labels.csv

Keyboard shortcuts:
    0-9      : Number symbols (0 through 9)
    T        : +2 (draw Two)
    F        : +4 (draw Four)
    W        : Wild
    S        : Skip
    R        : Reverse
    Space    : Flag as NOT USABLE (kept in CSV but excluded from training)
    Z        : Undo the last label
    Esc      : Quit (everything is auto-saved as you go)

Output: A CSV with columns [filename, label, usable]. Re-running resumes
where you left off — only unlabeled images are shown.

Dependencies:
    pip install Pillow
    (tkinter ships with Python; on some Linux distros: sudo apt install python3-tk)
"""

import argparse
import csv
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from PIL import Image, ImageTk

# Map a single keypress to a label string.
KEY_TO_LABEL = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "t": "+2",
    "f": "+4",
    "w": "wild",
    "s": "skip",
    "r": "reverse",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

DISPLAY_SIZE = 400  # Up-scaled view of the 80x80 crop


class LabelerApp:
    def __init__(self, root, image_dir, label_file):
        self.root = root
        self.image_dir = Path(image_dir)
        self.label_file = Path(label_file)

        # filename -> (label, usable)
        self.labels = self._load_labels()

        self.all_images = sorted(
            p.name
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self.queue = [f for f in self.all_images if f not in self.labels]
        self.history = []  # stack of filenames in the order they were labeled

        self._build_ui()
        self._bind_keys()
        self._show_current()

    # ---------- persistence ----------

    def _load_labels(self):
        labels = {}
        if self.label_file.exists():
            with open(self.label_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labels[row["filename"]] = (row["label"], row["usable"])
        return labels

    def _save_all(self):
        # Atomic-ish: write to temp then replace.
        tmp = self.label_file.with_suffix(self.label_file.suffix + ".tmp")
        with open(tmp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label", "usable"])
            for fn, (lbl, us) in self.labels.items():
                writer.writerow([fn, lbl, us])
        os.replace(tmp, self.label_file)

    # ---------- UI ----------

    def _build_ui(self):
        self.root.title("UNO Card Symbol Labeler")
        self.root.configure(bg="white")

        self.progress_var = tk.StringVar()
        tk.Label(
            self.root,
            textvariable=self.progress_var,
            font=("Arial", 14, "bold"),
            bg="white",
        ).pack(pady=(10, 5))

        self.image_label = tk.Label(self.root, bg="white")
        self.image_label.pack(pady=5)

        self.filename_var = tk.StringVar()
        tk.Label(
            self.root,
            textvariable=self.filename_var,
            font=("Arial", 10),
            fg="gray",
            bg="white",
        ).pack()

        self.recent_var = tk.StringVar()
        tk.Label(
            self.root,
            textvariable=self.recent_var,
            font=("Arial", 10),
            fg="#0066aa",
            bg="white",
        ).pack(pady=5)

        # Class buttons grid
        grid = tk.Frame(self.root, bg="white")
        grid.pack(pady=10)

        layout = [
            [("0", "0"), ("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")],
            [("5", "5"), ("6", "6"), ("7", "7"), ("8", "8"), ("9", "9")],
            [
                ("+2  (T)", "t"),
                ("+4  (F)", "f"),
                ("Wild  (W)", "w"),
                ("Skip  (S)", "s"),
                ("Reverse  (R)", "r"),
            ],
        ]
        for r, row in enumerate(layout):
            for c, (text, key) in enumerate(row):
                tk.Button(
                    grid,
                    text=text,
                    width=12,
                    height=2,
                    font=("Arial", 11),
                    command=lambda k=key: self._label(k),
                ).grid(row=r, column=c, padx=3, pady=3)

        special = tk.Frame(self.root, bg="white")
        special.pack(pady=8)
        tk.Button(
            special,
            text="Not Usable  (Space)",
            width=20,
            height=2,
            bg="#ffd0d0",
            font=("Arial", 11),
            command=self._mark_unusable,
        ).grid(row=0, column=0, padx=3)
        tk.Button(
            special,
            text="Undo  (Z)",
            width=14,
            height=2,
            font=("Arial", 11),
            command=self._undo,
        ).grid(row=0, column=1, padx=3)
        tk.Button(
            special,
            text="Quit  (Esc)",
            width=14,
            height=2,
            font=("Arial", 11),
            command=self._quit,
        ).grid(row=0, column=2, padx=3)

    def _bind_keys(self):
        for key in KEY_TO_LABEL:
            self.root.bind(key, lambda e, k=key: self._label(k))
            self.root.bind(key.upper(), lambda e, k=key: self._label(k))
        self.root.bind("<space>", lambda e: self._mark_unusable())
        self.root.bind("z", lambda e: self._undo())
        self.root.bind("Z", lambda e: self._undo())
        self.root.bind("<Escape>", lambda e: self._quit())

    def _show_current(self):
        total = len(self.all_images)
        done = len(self.labels)
        usable = sum(1 for _, u in self.labels.values() if u == "1")
        flagged = done - usable

        self.progress_var.set(
            f"Labeled: {done} / {total}   "
            f"(usable: {usable}, flagged: {flagged})   "
            f"Remaining: {len(self.queue)}"
        )

        if not self.queue:
            self.filename_var.set("✓ All images labeled!")
            self.image_label.config(image="")
            self.recent_var.set(
                f"Done. Usable: {usable}, flagged: {flagged}. "
                f"Output: {self.label_file}"
            )
            return

        current = self.queue[0]
        self.filename_var.set(current)

        try:
            img = Image.open(self.image_dir / current).convert("RGB")
            img_disp = img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_disp)
            self.image_label.config(image=self.photo)
        except Exception as e:
            print(f"[warning] failed to load {current}: {e}")
            self.queue.pop(0)
            self._show_current()
            return

        if self.history:
            last_five = self.history[-5:]
            parts = []
            for fn in last_five:
                lbl, us = self.labels[fn]
                shown = lbl if us == "1" else "✗"
                parts.append(f"{fn[:12]}→{shown}")
            self.recent_var.set("Recent:  " + "   ".join(parts))
        else:
            self.recent_var.set("")

    # ---------- actions ----------

    def _commit(self, filename, label, usable):
        self.labels[filename] = (label, usable)
        self.history.append(filename)
        self._save_all()

    def _label(self, key):
        if not self.queue:
            return
        label = KEY_TO_LABEL[key]
        current = self.queue.pop(0)
        self._commit(current, label, "1")
        self._show_current()

    def _mark_unusable(self):
        if not self.queue:
            return
        current = self.queue.pop(0)
        self._commit(current, "", "0")
        self._show_current()

    def _undo(self):
        if not self.history:
            return
        last = self.history.pop()
        self.labels.pop(last, None)
        self.queue.insert(0, last)
        self._save_all()
        self._show_current()

    def _quit(self):
        self._save_all()
        self.root.quit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image_dir", help="Directory containing UNO card corner images")
    p.add_argument(
        "--labels",
        default="labels.csv",
        help="CSV file to read/write labels (default: labels.csv)",
    )
    args = p.parse_args()

    if not os.path.isdir(args.image_dir):
        print(f"Error: {args.image_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    LabelerApp(root, args.image_dir, args.labels)
    root.mainloop()


if __name__ == "__main__":
    main()
