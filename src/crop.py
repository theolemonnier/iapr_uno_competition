import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from src.utils import open_image
from src.detect_reference import extract_color_masks, plot_one_image

def draw_box(img, x, y, w, h): 

    fig, ax = plt.subplots()

    ax.imshow(img)

    rect = patches.Rectangle(
        (x, y),     # top-left corner
        w,           # width
        h,           # height
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )

    ax.add_patch(rect)

    plt.show()

def save_crop(img, x, y, w, h, title): 
    crop = img[y:y+h, x:x+w]
    cv2.imwrite(title, crop)

if __name__ == '__main__': 

    # choose an image
    selection = "reference" # "train_white_bg", "train_colorful_bg"
    img = open_image(selection=selection)

    # exctract color channels (Red, Blue, Green, Yellow, Black)
    masks = extract_color_masks(img=img, plot=False)

    # choose which color you want
    colour = masks["Red"] # Blue, Green, Yellow, Black

    # choose box params
    x = 2270
    y = 700
    w = 60
    h = 80 

    # draw_box
    draw_box(img=colour, x=x, y=y, w=w, h=h)

    # change title and decomment to save the crop
    # save_crop(img=colour, x=x, y=y, w=w, h=h)


