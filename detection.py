import cv2
import matplotlib.pyplot as plt
import pydicom as dicom
from pandas import np
from pydicom import dcmread
from scipy.sparse import csc_matrix


path = 'C:\\Users\\bzavo\\PycharmProjects\\MainProject\\DICOM\\'
image = 'IMG0000000100.dcm'
# filename = 'RTSS.dcm'
# ds = dcmread(path + filename)
img = dcmread(path + image)

plt.imsave('imagetotal.png', img.pixel_array) #saving the initial image
img = cv2.imread('imagetotal.png', 0)

#upscale the image for better processing
scale_percent = 1000  # percent of the original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resizing the image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
img = img[1250:4000, 1100:3800]
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
hsv = cv2.cvtColor(cimg, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (150, 150, 150), (190, 190, 190))


circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 200,
                            param1 = 1, param2 = 8, minRadius = 20, maxRadius = 30)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

# cv2.imshow('detected circles', cimg)
plt.imshow(cimg)
plt.show()
cv2.imwrite('detectedcircles.png', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


