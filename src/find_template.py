import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from src.utils import open_image
from src.detect_reference import extract_color_masks, plot_one_image
from config import REFERENCE_IMAGES
import numpy as np


def template_matching(): 
    # choose an image
    selection = "train_colorful_bg" # "train_white_bg", "train_colorful_bg"
    img = open_image(selection=selection)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # exctract color channels (Red, Blue, Green, Yellow, Black)
    masks = extract_color_masks(img=img, plot=False)

    # choose which color you want
    img_binary = masks["Red"] # Blue, Green, Yellow, Black

    # template 
    crop_9 = cv2.imread(REFERENCE_IMAGES / "crop_9.jpg", cv2.IMREAD_GRAYSCALE)
    crop_9 = cv2.rotate(crop_9, cv2.ROTATE_90_COUNTERCLOCKWISE)
    height, width = crop_9.shape

    plot_one_image(crop_9, title="template_9")


    methods = [cv2.TM_CCOEFF, 
               cv2.TM_CCOEFF_NORMED, 
               cv2.TM_CCORR,
               cv2.TM_SQDIFF, 
               cv2.TM_SQDIFF_NORMED]
    
    titles = ["cv2.TM_CCOEFF", 
               "cv2.TM_CCOEFF_NORMED", 
               "cv2.TM_CCORR",
               "cv2.TM_SQDIFF", 
               "cv2.TM_SQDIFF_NORMED"]
    
    for i in range(len(methods)): 
        cur_img = img_rgb.copy()
        template_map = cv2.matchTemplate(img_binary, crop_9, methods[i])

        min, max, min_loc, max_loc = cv2.minMaxLoc(template_map)
        
        if methods[i] == cv2.TM_SQDIFF or methods[i] == cv2.TM_SQDIFF_NORMED: 
            top_left = min_loc
            print(f"min is {min}")
        else: 
            top_left = max_loc
            print(f"max is : {max}")

        bottom_right = (top_left[0] + width, top_left[1]+height)
        cv2.rectangle(cur_img, top_left, bottom_right, (255, 255, 255), 10)
        plt.figure()
        plt.subplot(121)
        plt.imshow(template_map)
        plt.title(titles[i])
        plt.subplot(122)
        plt.imshow(cur_img)

    plt.show()

if __name__ == "__main__": 
    template_matching()
