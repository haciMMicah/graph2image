import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def main():
    img = cv.imread('../resources/witcherMedallion.jpg')
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    im3 = np.copy(img)
    cv.drawContours(im3, contours, -1, (0, 255, 0), 3)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(im3, cmap='gray')
    plt.title('Contours'), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    main()
