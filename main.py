from ast import Load

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

# directory of the current python file
BASE_DIR = Path(__file__).resolve().parent

img_path = BASE_DIR / "reference_images" / "L1000766.jpg"

# img_path = BASE_DIR / "train_images" / "L1000777.jpg" # white bg 
# img_path = BASE_DIR / "train_images" / "L1000974.jpg" # colorfull bg
# img_path = BASE_DIR / "train_images" / "L1000836.jpg" # white bg and 2 identical colors 

img = cv2.imread(str(img_path))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

H, S, V = cv2.split(hsv)

# plt.figure(figsize=(15,5))

# plt.subplot(1,3,1)
# plt.hist(H.ravel(), bins=180, range=[0,180])
# plt.title("Hue Histogram")

# plt.subplot(1,3,2)
# plt.hist(S.ravel(), bins=256, range=[0,256])
# plt.title("Saturation Histogram")

# plt.subplot(1,3,3)
# plt.hist(V.ravel(), bins=256, range=[0,256])
# plt.axvline(x=210, color='r', linestyle='--')

# plt.title("Value Histogram")

# plt.show()

############## threshold on value ###############
# # Threshold on Value channel
# threshold = 210

# # keep dark objects (cards) against bright table
# mask = (V < threshold).astype("uint8") * 255

# # Display
# plt.figure(figsize=(12,5))

# plt.subplot(1,2,1)
# plt.imshow(rgb)
# plt.title("Original")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(mask, cmap="gray")
# plt.title(f"V < {threshold}")
# plt.axis("off")

# plt.show()

# ############## threshold on gray scale ###############
# # Load grayscale image
# gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

# # Plot image + histogram
# plt.figure(figsize=(14,5))

# # Grayscale image
# plt.subplot(1,2,1)
# plt.imshow(gray, cmap="gray")
# plt.title("Grayscale Image")
# plt.axis("off")

# # Histogram
# plt.subplot(1,2,2)
# plt.hist(gray.ravel(), bins=256, range=[0,256])

# threshold = 210
# plt.axvline(x=threshold, color='r', linestyle='--',
#             label=f'threshold = {threshold}')

# plt.title("Grayscale Histogram")
# plt.xlabel("Gray level")
# plt.ylabel("Pixel count")
# plt.xlim([0,256])
# plt.legend()
# plt.grid()

# plt.show()

##### separate all colors with a single threshold on gray scale ###############

img = cv2.imread(str(img_path))
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV ranges in OpenCV:
# H: 0-179, S: 0-255, V: 0-255

color_ranges = {
    "Yellow": ((20, 60, 80), (40, 255, 255)),
    "Red 1":  ((0, 60, 60),  (10, 255, 255)),
    "Red 2":  ((170, 60, 60), (179, 255, 255)),
    "Blue":   ((85, 50, 50),  (120, 255, 255)),
    "Green":  ((40, 40, 50),  (85, 255, 255)),
    "Black":  ((0, 0, 0),     (179, 255, 80)),
}

masks = {}

# Yellow
masks["Yellow"] = cv2.inRange(hsv, np.array(color_ranges["Yellow"][0]), np.array(color_ranges["Yellow"][1]))

# Red needs two hue intervals
red1 = cv2.inRange(hsv, np.array(color_ranges["Red 1"][0]), np.array(color_ranges["Red 1"][1]))
red2 = cv2.inRange(hsv, np.array(color_ranges["Red 2"][0]), np.array(color_ranges["Red 2"][1]))
masks["Red"] = cv2.bitwise_or(red1, red2)

# --- BLACK MASK WITH RGB ---

b, g, r = cv2.split(img)

threshold = 150

black_mask = (
    (r < threshold) &
    (g < threshold) &
    (b < threshold)
).astype(np.uint8) * 255

masks["Black"] = black_mask

for color in ["Blue", "Green", "Black"]:
    lower, upper = color_ranges[color]
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

    if color == "Black":
        # Option A: black pixels appear white, everything else black
        masks[color] = black_mask

        # Option B: black pixels appear black, everything else white
        # masks[color] = cv2.bitwise_not(mask)

    else:
        masks[color] = mask

# plot images 
plt.figure(figsize=(18, 8))

for i, (name, mask) in enumerate(masks.items(), start=1):
    plt.subplot(1, 5, i)

    if name == "Black":
        plt.imshow(mask, cmap="gray")
    else:
        filtered = cv2.bitwise_and(rgb, rgb, mask=mask)
        plt.imshow(filtered)

    plt.title(name)
    plt.axis("off")
plt.show()


# # create dico 
# filtered_images = {}
# for name, mask in masks.items():

#     if name == "Black":
#         # Keep black mask as binary image
#         filtered_images[name] = mask.copy()

#     else:
#         filtered = cv2.bitwise_and(rgb, rgb, mask=mask)
#         filtered_images[name] = filtered


# ###########  find geometry of cards ###############
# import cv2
# import numpy as np

# img = filtered_images["Green"]



# # img = cv2.imread(img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # Threshold white cards
# _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# plt.figure(figsize=(10,5))
# plt.imshow(th, cmap="gray")
# plt.show()

# contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cards = []

# img_draw = th.copy()

# for cnt in contours:
#     area = cv2.contourArea(cnt)
#     if area < 1:
#         continue

#     peri = cv2.arcLength(cnt, True)
#     approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

#     if len(approx) == 4:
#         cv2.drawContours(img_draw, [approx], -1, (0,255,0), 3)

# # OpenCV uses BGR, matplotlib expects RGB
# img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(8,10))
# plt.imshow(img_rgb)
# plt.axis('off')
# plt.show()