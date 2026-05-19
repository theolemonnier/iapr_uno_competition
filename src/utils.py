import cv2
from pathlib import Path


def plot_one_image(img, title="Image"):
    import matplotlib.pyplot as plt
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 uses BGR; convert to RGB for matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def open_image(path):
    """
    Open an image from a file path.

    Parameters
    ----------
    path : str or Path
        Path to the image file.

    Returns
    -------
    img : np.ndarray
        Loaded image in BGR format.
    """
    path = Path(path)
    print(f"Processing image: {path}")
    img = cv2.imread(str(path))
    return img
