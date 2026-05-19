import cv2
import numpy as np
from src.utils import open_image
from src.detect_reference import extract_color_masks, plot_one_image


def template_matching(selection, reference_images_path, color="Red"):
    """
    Run template matching on a scene image using a reference card crop.

    Parameters
    ----------
    selection : str or Path
        Path to the scene image.
    reference_images_path : Path
        Directory containing reference card crops.
    color : str
        Color channel to match against (default "Red").
    """
    import matplotlib.pyplot as plt

    img = open_image(path=selection)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks = extract_color_masks(img=img, plot=False)
    img_binary = masks[color]

    crop_9 = cv2.imread(str(reference_images_path / "crop_9.jpg"), cv2.IMREAD_GRAYSCALE)
    crop_9 = cv2.rotate(crop_9, cv2.ROTATE_90_COUNTERCLOCKWISE)
    height, width = crop_9.shape

    plot_one_image(crop_9, title="template_9")

    methods = [
        cv2.TM_CCOEFF,
        cv2.TM_CCOEFF_NORMED,
        cv2.TM_CCORR,
        cv2.TM_SQDIFF,
        cv2.TM_SQDIFF_NORMED,
    ]
    titles = [
        "cv2.TM_CCOEFF",
        "cv2.TM_CCOEFF_NORMED",
        "cv2.TM_CCORR",
        "cv2.TM_SQDIFF",
        "cv2.TM_SQDIFF_NORMED",
    ]

    for i in range(len(methods)):
        cur_img = img_rgb.copy()
        template_map = cv2.matchTemplate(img_binary, crop_9, methods[i])

        mn, mx, min_loc, max_loc = cv2.minMaxLoc(template_map)

        if methods[i] in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            top_left = min_loc
            print(f"min is {mn}")
        else:
            top_left = max_loc
            print(f"max is: {mx}")

        bottom_right = (top_left[0] + width, top_left[1] + height)
        cv2.rectangle(cur_img, top_left, bottom_right, (255, 255, 255), 10)

        plt.figure()
        plt.subplot(121)
        plt.imshow(template_map)
        plt.title(titles[i])
        plt.subplot(122)
        plt.imshow(cur_img)

    plt.show()

# Removed: exploratory __main__ block (see git history).
