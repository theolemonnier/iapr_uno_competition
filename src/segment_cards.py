from ast import Load
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# import from config file
from config import REFERENCE_IMAGES
from config import IMG_PATHS

# imports frim utils 
from src.utils import open_image, plot_one_image

if __name__ == "__main__":
    selection = "train_white_overlapp" # "train_white_bg", "train_colorful_bg"
    img = open_image(selection=selection)
    # plot_one_image(img)

    # hls
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # plot_one_image(s, title="s")

    # add blur 
    blurred = cv2.GaussianBlur(s, (5, 5), 0)
    plot_one_image(blurred, title="blurred")

    # Load image directly in grayscale

    # Compute histogram
    hist = cv2.calcHist(
        [blurred],
        [0],        # grayscale channel
        None,
        [256],      # bins
        [0,256]
    )
    threshold_min = 30
    threshold_max = 255 # 100

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(hist)
    plt.xlim([0,256])

    plt.xlabel("Pixel intensity")
    plt.ylabel("Number of pixels")
    plt.axvline(x=threshold_min)
    plt.title("Grayscale histogram")
    plt.show()

    # Mask: channel0 < 4
    mask = cv2.inRange(blurred, threshold_min, threshold_max)

    # Small kernel
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5,5)
    )

    # Closing = dilation + erosion
    closed_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
)

##########
    contours, _ = cv2.findContours(
        closed_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []

    out = cv2.cvtColor(closed_mask, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40000:
            continue
        perimeter = cv2.arcLength(cnt, True)

        approx = cv2.approxPolyDP(
            cnt,
            0.03 * perimeter,   # tune this
            True
        )

        n_vertices = len(approx)

        # Keep triangles, rectangles, and polygons
        if n_vertices >= 3:
            polygons.append(approx)

            if n_vertices == 3:
                label = "triangle"
                color = (255, 0, 0)
            elif n_vertices == 4:
                label = "rectangle/quad"
                color = (0, 255, 0)
            else:
                label = f"{n_vertices}-polygon"
                color = (0, 0, 255)

            cv2.drawContours(out, [approx], -1, color, 2)

            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(
                out,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Detected polygons", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
