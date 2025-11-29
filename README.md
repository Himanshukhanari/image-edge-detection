This project demonstrates edge detection on any input image using four widely used operators:

Sobel

Prewitt

Laplacian

Canny

The results are displayed side-by-side for easy comparison using OpenCV, NumPy, SciPy, and Matplotlib.
│── edge_detection.py
│── image.jpg
│── README.md
Edge detection is a fundamental task in image processing used to detect discontinuities or sharp changes in pixel intensity.
This project helps visualize how different operators detect edges uniquely.
code used
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ----------- Load Image -----------
img = cv2.imread("image.jpg")      # <-- replace with your image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----------- Sobel Operator -----------
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobel_x, sobel_y)

# ----------- Prewitt Operator -----------
prewitt_x = ndimage.convolve(img_gray, np.array([[1,0,-1],[1,0,-1],[1,0,-1]]))
prewitt_y = ndimage.convolve(img_gray, np.array([[1, 1, 1],[0, 0, 0],[-1,-1,-1]]))
prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)

# ----------- Laplacian Operator -----------
laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)

# ----------- Canny Edge Detection -----------
canny = cv2.Canny(img_gray, 100, 200)

# ----------- Plot Results -----------
titles = ["Original", "Sobel", "Prewitt", "Laplacian", "Canny"]
images = [img_gray, sobel, prewitt, laplacian, canny]

plt.figure(figsize=(14, 8))
for i in range(5):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")

plt.tight_layout()
plt.show()
Operators Explained
1. Sobel Operator

Detects edges using intensity gradients in x and y directions.
Best for detecting strong edges while reducing noise.

2. Prewitt Operator

Similar to Sobel but uses simpler kernels.
Good for educational purposes and basic edge detection.

3. Laplacian Operator

Second-order derivative operator that detects rapid intensity changes.
Useful for highlighting fine edges.

4. Canny Edge Detector

A multi-stage, advanced edge detector using:

Gaussian smoothing

Gradient computation

Non-max suppression

Hysteresis thresholding

Produces the cleanest and sharpest edges.
