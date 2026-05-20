import cv2
import numpy as np
from pathlib import Path


def plot_one_image(img, title="Image"):
    import matplotlib.pyplot as plt
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 uses BGR; convert to RGB for matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

# Removed: exploratory __main__ block (see git history).
