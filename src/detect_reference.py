from ast import Load
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
from typing import Tuple, Optional


# import from config file
from config import REFERENCE_IMAGES
from config import IMG_PATHS

# img selection
img_selected = "reference" # change this to select different image
path_to_img = IMG_PATHS[img_selected]

print(f"Processing image: {path_to_img}")

def plot_one_image(img, title="Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

def hsv_mask(hsv, lower, upper):
    return cv2.inRange(
        hsv,
        np.array(lower),
        np.array(upper)
    )

def extract_color_masks(img_path, plot=False):
    """
    Extract color masks from an image.

    Parameters
    ----------
    img_path : str or Path
        Path to image.
    plot : bool, optional
        If True, display extracted masks and filtered images.

    Returns
    -------
    masks : dict
        Dictionary containing masks:
        {
            "Yellow": ...,
            "Red": ...,
            "Blue": ...,
            "Green": ...,
            "Black": ...
        }
    """

    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV ranges
    color_ranges = {
        "Yellow": ((20, 60, 80), (40, 255, 255)),
        "Red 1":  ((0, 60, 60), (10, 255, 255)),
        "Red 2":  ((170, 60, 60), (179, 255, 255)),
        "Blue":   ((85, 50, 50), (120, 255, 255)),
        "Green":  ((40, 40, 50), (85, 255, 255)),
        "Black":  ((0, 0, 0), (179, 255, 80)),
    }

    masks = {}

    # Yellow
    lower, upper = color_ranges["Yellow"]
    masks["Yellow"] = cv2.inRange(
        hsv,
        np.array(lower),
        np.array(upper)
    )

    # Red (two hue ranges)
    red1 = cv2.inRange(
        hsv,
        np.array(color_ranges["Red 1"][0]),
        np.array(color_ranges["Red 1"][1])
    )

    red2 = cv2.inRange(
        hsv,
        np.array(color_ranges["Red 2"][0]),
        np.array(color_ranges["Red 2"][1])
    )

    masks["Red"] = cv2.bitwise_or(red1, red2)

    # Black using RGB thresholding
    b, g, r = cv2.split(img)

    threshold = 150

    black_mask = (
        (r < threshold) &
        (g < threshold) &
        (b < threshold)
    ).astype(np.uint8) * 255

    masks["Black"] = black_mask

    # Blue and Green
    for color in ["Blue", "Green"]:
        lower, upper = color_ranges[color]

        masks[color] = cv2.inRange(
            hsv,
            np.array(lower),
            np.array(upper)
        )

    # Optional plotting
    if plot:
        plt.figure(figsize=(18, 8))

        for i, (name, mask) in enumerate(masks.items(), start=1):
            plt.subplot(1, len(masks), i)

            if name == "Black":
                plt.imshow(mask, cmap="gray")
            else:
                filtered = cv2.bitwise_and(
                    rgb,
                    rgb,
                    mask=mask
                )
                plt.imshow(filtered)

            plt.title(name)
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    return masks

def detect_contours(binary_img, min_area=1000, plot=False):
    edges = cv2.Canny(binary_img, 100, 255)

    # plot_one_image(edges, title="Canny edges")
    # Close small gaps (recommended)
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    # plot_one_image(edges, title="Closed edges")

    # Find contours
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Keep only large contours
    min_area = 2500   # adjust this 25000
    large_contours = [
        cnt for cnt in contours
        if cv2.contourArea(cnt) > min_area
    ]

    # Draw result
    result = np.zeros_like(binary_img)

    cv2.drawContours(
        result,
        large_contours,
        -1,
        255,
        thickness=2
    )

    # Display with matplotlib
    if plot == True:
        plt.figure(figsize=(8, 8))
        plt.imshow(result, cmap="gray")
        plt.title("Large contours")
        plt.axis("off")
        plt.show()

    # cv2.imwrite("contours_binary.png", result) if needed 

    return result

####### match template

def find_binary_template(template: np.ndarray, image: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Check if a binary template (white part only) exists in a binary image.
    
    Parameters:
    -----------
    template : np.ndarray
        Binary image template to search for (0=black, 255=white or True/False)
    image : np.ndarray
        Binary image to search in (0=black, 255=white or True/False)
    
    Returns:
    --------
    Optional[Tuple[int, int]]
        (row, col) position of top-left corner if found, None otherwise
    
    How it works:
    -------------
    1. Slides the template over the image at every position
    2. At each position, checks if all WHITE pixels in template match
    3. Returns the first position where template matches
    """
    # Convert to binary (True/False) if needed
    template = template > 0
    image = image > 0
    
    template_height, template_width = template.shape
    image_height, image_width = image.shape
    
    # Template must fit in image
    if template_height > image_height or template_width > image_width:
        return None
    
    # Slide template over image
    for row in tqdm(range(image_height - template_height + 1)):
        for col in range(image_width - template_width + 1):
            # Extract the region of interest from the image
            roi = image[row:row + template_height, col:col + template_width]
            
            # Check if all white pixels in template match the ROI
            # We only care about white pixels (True) in template
            match = np.all(template == roi, where=template)
            
            if match:
                return (row, col)
    
    return None

def visualize_detections(image, detections, template_edges=None):
    """
    Visualize detected cards on the original image.
    
    Parameters
    ----------
    image : np.ndarray
        Original RGB or grayscale image.
    detections : list of dict
        List of detections from match_template_multiscale_rotation.
    template_edges : np.ndarray, optional
        Template edge image to display.
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        vis = image.copy()
    
    # Draw detections
    for i, det in enumerate(detections):
        box = det['box']
        score = det['score']
        angle = det['angle']
        
        # Draw bounding box
        cv2.drawContours(vis, [box], -1, (0, 255, 0), 3)
        
        # Draw center point
        center = (
            int(round(det['center'][0])),
            int(round(det['center'][1]))
        )

        cv2.circle(vis, center, 5, (255, 0, 0), -1)
        
        # Add label
        label = f"#{i+1} s={score:.2f} a={angle}°"
        cv2.putText(
            vis,
            label,
            (center[0] - 50, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )
    
    # Plot
    fig, axes = plt.subplots(1, 2 if template_edges is not None else 1, 
                             figsize=(15, 8))
    
    if template_edges is not None:
        axes[0].imshow(template_edges, cmap='gray')
        axes[0].set_title('Reference Template')
        axes[0].axis('off')
        
        axes[1].imshow(vis)
        axes[1].set_title(f'Detections: {len(detections)} cards found')
        axes[1].axis('off')
    else:
        axes.imshow(vis)
        axes.set_title(f'Detections: {len(detections)} cards found')
        axes.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return vis


def overlay_template(target, template, x=50, y=50, alpha=0.6):
    img = target.copy()

    # Ensure target is BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Ensure template is grayscale mask
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template

    h, w = template_gray.shape[:2]

    if y + h > img.shape[0] or x + w > img.shape[1]:
        raise ValueError("Template exceeds image bounds")

    colored = np.zeros((h, w, 3), dtype=np.uint8)
    colored[:, :, 2] = template_gray  # red overlay

    roi = img[y:y+h, x:x+w]

    mask = template_gray > 0
    roi[mask] = cv2.addWeighted(
        roi[mask],
        1 - alpha,
        colored[mask],
        alpha,
        0
    )

    img[y:y+h, x:x+w] = roi

    print(f"Template size: {w} x {h}")

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Template overlay")
    plt.axis("off")
    plt.show()

    return img


def angle_diff(a, b):
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)

def keep_perpendicular_lines(lines, angle_tol=10):
    """
    Keep only lines that are perpendicular to at least one other line.
    angle_tol is in degrees.
    """
    if lines is None:
        return []

    parsed = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        parsed.append({
            "line": line,
            "angle": angle,
            "length": length
        })

    kept = []

    for i, l1 in enumerate(parsed):
        has_perpendicular = False

        for j, l2 in enumerate(parsed):
            if i == j:
                continue

            diff = angle_diff(l1["angle"], l2["angle"])

            if abs(diff - 90) < angle_tol:
                has_perpendicular = True
                break

        if has_perpendicular:
            kept.append(l1["line"])

    return kept

def detect_card_lines(edge_img, original_img=None, plot=True):
    # Make sure image is uint8 binary
    edges = (edge_img > 0).astype(np.uint8) * 255
    kernel = np.ones((7,7), np.uint8)

    edges = cv2.dilate(edges, kernel, iterations=1)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=280,
        minLineLength=50,
        maxLineGap=15
    )

    lines = keep_perpendicular_lines(lines, angle_tol=10)
    if original_img is None:
        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        vis = original_img.copy()

    detected_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            detected_lines.append({
                "p1": (x1, y1),
                "p2": (x2, y2),
                "length": length,
                "angle": angle
            })

            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print(f"Detected {len(detected_lines)} line segments")

    if plot:
        plt.figure(figsize=(10, 8))
        if original_img is None:
            plt.imshow(vis, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title("Hough line segments")
        plt.axis("off")
        plt.show()

    return detected_lines, vis

if __name__ == "__main__":


    filtered_images = extract_color_masks(path_to_img, plot=False)

    binary_img = filtered_images["Red"] # change this to select different color
    plot_one_image(binary_img)

    # Elliptical kernel usually works better for natural shapes
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (4, 4)
    )

    # Close gaps
    closed = cv2.morphologyEx(
        binary_img,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    cv2.imshow("closed", closed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # plot_one_image(binary_img, title="Grayscale image")
      
    # Detect edges
    card_contours = contours_img = detect_contours(binary_img, min_area=25000, plot=False)



#     # Load binary image
#     img = card_contours

#     # Find contours
#     contours, _ = cv2.findContours(
#         img,
#         cv2.RETR_EXTERNAL,
#         cv2.CHAIN_APPROX_SIMPLE
#     )

#     out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#     for cnt in contours:

#         # Ignore tiny contours
#         if cv2.contourArea(cnt) < 100:
#             continue

#         # Approximate contour
#         epsilon = 0.02 * cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)

#         pts = approx[:,0,:]
#         n = len(pts)

#         if n < 3:
#             continue

#         # Check each vertex
#         for i in range(n):

#             prev = pts[(i-1)%n]
#             curr = pts[i]
#             nxt  = pts[(i+1)%n]

#             # vectors
#             v1 = prev - curr
#             v2 = nxt - curr

#             # normalize
#             v1 = v1 / np.linalg.norm(v1)
#             v2 = v2 / np.linalg.norm(v2)

#             # angle
#             angle = np.degrees(
#                 np.arccos(np.clip(np.dot(v1,v2), -1.0, 1.0))
#             )

#             # Keep near 90°
#             if 80 <= angle <= 100:

#                 x, y = curr

#                 # Draw detected right angle
#                 cv2.circle(out, (x,y), 6, (0,0,255), -1)

#                 print(f"Right angle at ({x},{y}) : {angle:.1f}")

#     cv2.imshow("Right angles", out)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()







#     # plot_one_image(card_contours, title="Card contours")

#     # my_ref = cv2.imread(str(IMG_PATHS["cards_contours_reference"]), cv2.IMREAD_GRAYSCALE)
#     # # plot_one_image(my_ref, title="Reference contours")


#     # target_edges = card_contours
#     # template_edges = my_ref

#     # original_img = cv2.imread(str(path_to_img))


# #     lines, line_vis = detect_card_lines(
# #         edge_img=card_contours,
# #         original_img=original_img,
# #         plot=True
# # )



#     # 3. Match template
#     # # Search for template
#     # result = find_binary_template(template_edges, target_edges)
    
#     # if result:
#     #     print(f"✓ Template found at position: {result}")
#     #     print(f"  Row: {result[0]}, Column: {result[1]}")
#     # else:
#     #     print("✗ Template not found")
    
#     # # 4. Visualize results
#     # original_img = cv2.imread(str(path_to_img))
#     # original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
#     # visualize_detections(original_rgb, detections, template_edges)
    
#     # # Print detection details
#     # print("\nDetection details:")
#     # for i, det in enumerate(detections):
#     #     print(f"Card {i+1}:")
#     #     print(f"  Center: {det['center']}")
#     #     print(f"  Angle: {det['angle']}°")
#     #     print(f"  Scale: {det['scale']}")
#     #     print(f"  Score: {det['score']:.3f}")


#     # # scale = 2.5

#     # # template_scaled = cv2.resize(
#     # #     my_ref,
#     # #     None,
#     # #     fx=scale,
#     # #     fy=scale,
#     # #     interpolation=cv2.INTER_NEAREST
#     # # )

#     # # overlay_template(target_edges, template_scaled, x=100, y=100)