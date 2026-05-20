import cv2
import sys


def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        img, scale = params
        # Map display coordinates back to original image coordinates
        ox, oy = int(x / scale), int(y / scale)
        b, g, r = img[oy, ox]
        h, s, v = cv2.cvtColor(img[oy : oy + 1, ox : ox + 1], cv2.COLOR_BGR2HSV)[0, 0]
        print(
            f"({ox:4d}, {oy:4d})  RGB=({r:3d}, {g:3d}, {b:3d})  "
            f"HSV=({h:3d}, {s:3d}, {v:3d})"
        )


def fit_to_screen(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)  # never upscale
    if scale < 1.0:
        img_disp = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        img_disp = img
    return img_disp, scale


def main():
    if len(sys.argv) < 2:
        print("Usage: python pixel_picker.py <image_path>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not load image: {sys.argv[1]}")
        sys.exit(1)

    img_disp, scale = fit_to_screen(img)

    window = "Click pixels (press q or ESC to quit)"
    cv2.imshow(window, img_disp)
    cv2.setMouseCallback(window, on_click, (img, scale))

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
