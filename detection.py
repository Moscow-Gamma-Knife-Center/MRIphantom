import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydicom import dcmread

# open image from a folder
path = input("Enter the absolute path of the image:\n")
# path = 'C:\\Users\\bzavo\\PycharmProjects\\MainProject\\DICOM\\'
# image = 'IMG0000000110.dcm'
img = dcmread(path)

#save image of a slice and open it in opencv
plt.imsave('imagetotal.png', img.pixel_array) #saving the initial image
img = cv2.imread('imagetotal.png', 0)

#upscale the image for better processing
scale_percent = 1000  # percent of the original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

#blacken everything outside of ROI
myROI1 = [(0, 0), (0, 1375), (5115, 1375), (5115, 0)]
cv2.fillPoly(img, [np.array(myROI1)], 0)
myROI2 = [(0, 1375), (1255, 1375), (1255, 5119), (0, 5119)]
cv2.fillPoly(img, [np.array(myROI2)], 0)
myROI3 = [(0, 3839), (5115, 3839), (5115, 5119), (0, 5119)]
cv2.fillPoly(img, [np.array(myROI3)], 0)
myROI3 = [(0, 3839), (5115, 3839), (5115, 5119), (0, 5119)]
cv2.fillPoly(img, [np.array(myROI3)], 0)
myROI4 = [(3776, 0), (5115, 0), (5115, 5119), (3776, 5119)]
cv2.fillPoly(img, [np.array(myROI4)], 0)
myROI5 = [(2803, 1371), (2803, 1634), (5115, 1634), (5115, 1371)]
cv2.fillPoly(img, [np.array(myROI5)], 0)
myROI6 = [(3275, 1634), (3275, 1870), (5115, 1870), (5115, 1634)]
cv2.fillPoly(img, [np.array(myROI6)], 0)
myROI7 = [(3500, 1850), (3500, 2320), (5115, 2320), (5115, 1850)]
cv2.fillPoly(img, [np.array(myROI7)], 0)
myROI8 = [(3500, 2930), (3500, 3852), (5115, 3852), (5115, 2930)]
cv2.fillPoly(img, [np.array(myROI8)], 0)
myROI9 = [(3280, 3382), (3280, 3852), (5115, 3852), (5115, 3382)]
cv2.fillPoly(img, [np.array(myROI9)], 0)
myROI10 = [(2830, 3640), (2830, 3900), (5115, 3900), (5115, 3640)]
cv2.fillPoly(img, [np.array(myROI10)], 0)
myROI11= [(2180, 3640), (2180, 3900), (0, 3900), (0, 3640)]
cv2.fillPoly(img, [np.array(myROI11)], 0)
myROI11= [(1755, 3370), (1755, 3670), (0, 3670), (0, 3370)]
cv2.fillPoly(img, [np.array(myROI11)], 0)
myROI12= [(1500, 2900), (1500, 3400), (0, 3400), (0, 2900)]
cv2.fillPoly(img, [np.array(myROI12)], 0)
myROI13= [(1500, 2320), (1500, 1340), (0, 1340), (0, 2320)]
cv2.fillPoly(img, [np.array(myROI13)], 0)
myROI14= [(1740, 1840), (1740, 1200), (0, 1200), (0, 1840)]
cv2.fillPoly(img, [np.array(myROI14)], 0)
myROI15= [(2200, 1590), (2200, 1300), (0, 1300), (0, 1590)]
cv2.fillPoly(img, [np.array(myROI15)], 0)

cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
hsv = cv2.cvtColor(cimg, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (150, 150, 150), (190, 190, 190))

#detect circles
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 200,
                            param1 = 1, param2 = 8, minRadius = 20, maxRadius = 30)
circles = np.uint16(np.around(circles))

#draw circles, centers and their number
counter = 1
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    print(i[0] * 100 / scale_percent, i[1] * 100 / scale_percent)
    cv2.putText(cimg, f'{counter}', (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2, cv2.LINE_AA)
    counter += 1

plt.imshow(cimg)
plt.show()
#save processed images
cv2.imwrite('detectedcircles.png', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


