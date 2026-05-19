import cv2
from src.utils import open_image
from src.detect_reference import extract_color_masks, plot_one_image


def draw_box(img, x, y, w, h):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots()
    ax.imshow(img)

    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    plt.show()


def save_crop(img, x, y, w, h, title):
    crop = img[y:y+h, x:x+w]
    cv2.imwrite(title, crop)

# Removed: exploratory __main__ block (see git history).
