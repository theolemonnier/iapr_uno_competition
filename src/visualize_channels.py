from ast import Load
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# import from config file
from config import REFERENCE_IMAGES
from config import IMG_PATHS

def plot_one_image(img, title="Image"):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 does not use rgb but bgr, so we need to convert it to rgb for plotting with matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # img selection
    img_selected = "train_colorful_bg" # change this to select different image
    path_to_img = IMG_PATHS[img_selected]

    print(f"Processing image: {path_to_img}")

    img = cv2.imread(str(path_to_img))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # ycrcb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # hls
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h2, l2, s2 = cv2.split(hls)

    # edges_gray = cv2.Canny(gray, 50, 100)
    # edges_s = cv2.Canny(s, 10, 50)
    edges_v = cv2.Canny(v, 10, 80)


    channels = {
        "H": h,
        "S": s,
        "V": v,
        "L": l,
        "A": a,
        "B": b, 
        "Y": y, 
        "CR": cr, 
        "CB": cb, 
        "H2": h2,
        "L2": l2, 
        "S2": s2
    }

    for name, ch in channels.items(): 
        plot_one_image(ch, title=name)
