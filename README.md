# image-edge-detection
 perform edge detection using different operators and compare the results.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# ----------- Load Image -----------
img = cv2.imread("image.jpg")      # <-- change to your image name
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
