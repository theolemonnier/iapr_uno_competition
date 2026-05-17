import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image
import numpy as np
import cv2
import random

def display_one_image(img_name): 
    # Load annotations
    df = pd.read_csv("annotations.csv")
    df.columns = df.columns.str.strip()  # remove spaces in column names

    # load image
    img = Image.open("reference_annotations/"+img_name)
    
    # Plot image
    fig, ax = plt.subplots()
    ax.imshow(img)

    # Plot ellipses for this image
    for _, row in df[df["image"] == img_name].iterrows():
        ellipse = Ellipse(
            (
                row["posx"] + row["width"]/2,
                row["posy"] + row["height"]/2
            ),
            width=row["width"],
            height=row["height"],
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(ellipse)

    ax.set_axis_off()
    plt.show()

def findHullFromEllipse(posx, posy, width, height, n_points=50):
    """
    Create convex hull from a GIMP ellipse annotation.

    Inputs:
        posx, posy      : top-left corner of the ellipse bounding box
        width, height   : ellipse bounding box dimensions

    Returns:
        hull in OpenCV contour format: (N, 1, 2)
    """

    # Convert GIMP top-left coordinates to ellipse center
    cx = posx + width / 2
    cy = posy + height / 2

    a = width / 2
    b = height / 2

    points = []

    for t in np.linspace(0, 2*np.pi, n_points, endpoint=False):
        x = cx + a * np.cos(t)
        y = cy + b * np.sin(t)
        points.append([x, y])

    points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    return cv2.convexHull(points)

def draw_card_and_hull(rows): 
    rowHL = rows[rows["corner"] == "HL"].iloc[0]
    rowLR = rows[rows["corner"] == "LR"].iloc[0]

    # compute convex hull 
    hullHL = findHullFromEllipse(
        posx=rowHL.posx,
        posy=rowHL.posy,
        width=rowHL.width,
        height=rowHL.height
    )
    hullLR = findHullFromEllipse(
        posx=rowLR.posx,
        posy=rowLR.posy,
        width=rowLR.width,
        height=rowLR.height
    )

    # plot card_image with convex hulls    
    image_path = "data/card_images/"+card_name
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_plot = img.copy()
    cv2.drawContours(img_plot, [hullHL], -1, (255, 0, 0), 2)  # red
    cv2.drawContours(img_plot, [hullLR], -1, (0, 255, 0), 2)  # green
    plt.figure(figsize=(8, 8))
    plt.imshow(img_plot)
    plt.axis("off")
    plt.show()

if __name__ == '__main__': 
    random.seed(42)
    df = pd.read_csv("data/annotations.csv")

    # Pick a random image name from CSV
    card_name = random.choice(df["image"].unique())

    card_path = "data/card_images/" + card_name

    # Pick corresponding rows
    rows = df[df["image"] == card_name]

    print(f"Selected: {card_name}")
    print(rows)

    draw_card_and_hull(rows)
