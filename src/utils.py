import matplotlib.pyplot as plt
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

def open_image(selection): 
    # img selection
    img_selected = selection # change this to select different image
    path_to_img = IMG_PATHS[img_selected]

    print(f"Processing image: {path_to_img}")

    img = cv2.imread(str(path_to_img))

    return img