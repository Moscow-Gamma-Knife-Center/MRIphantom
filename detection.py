import cv2
import matplotlib.pyplot as plt
import numpy as np
from pydicom import dcmread
import math

def maskROI(img, modality):
    # detect ROI circle
    if "MR" in modality:
        ROIcircle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1000,
                                     param1=30, param2=40, minRadius=1000, maxRadius=1600)
        if ROIcircle is not None:
            ROIcircle = np.uint16(np.around(ROIcircle))
            mask = np.zeros_like(img)
            for i in ROIcircle[0, :]:
                # draw the outer circle
                mask = cv2.circle(mask, (i[0], i[1]), (i[2] - 30), (255, 255, 0), -1)
            maskedimg = cv2.bitwise_and(img, mask)
            return maskedimg
        else:
            print("No visible contours on this slice!")
            return None
    if "CT" in modality:
        ROIcircle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1000,
                                 param1=50, param2=40, minRadius=1000, maxRadius=1600)
        if ROIcircle is not None:
            ROIcircle = np.uint16(np.around(ROIcircle))
            mask = np.zeros_like(img)
            for i in ROIcircle[0, :]:
                # draw the outer circle
                mask = cv2.circle(mask, (i[0], i[1]), (i[2] - 80), (255, 255, 0), -1)
            maskedimg = cv2.bitwise_and(img, mask)
            return maskedimg
    else:
        print("No visible contours on this slice!")
        return None

def drawCircles(img, circles):
    # draw circles, centers and their number
    counter = 1
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
        # print(i[0] * 100 / scale_percent, i[1] * 100 / scale_percent)
        cv2.putText(img, f'{counter}', (int(i[0]), int(i[1])), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 2,
                    cv2.LINE_AA)
        counter += 1
    return img

def upscaleImage(img, scale_percent):
    # upscale the image for better processing
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def detectCircles(img, modality):
    if img is not None:
        if "MR" in modality:
            cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # detect circles
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 225,
                                       param1=20, param2=1, minRadius=20, maxRadius=28)
            circles = np.uint16(np.around(circles))
            print(circles.shape[1], " centers detected!")
            cimg = drawCircles(cimg, circles)
            return cimg, circles
        if "CT" in modality:
            cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # detect circles
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 225,
                                       param1=1, param2=8, minRadius=20, maxRadius=34)
            circles = np.uint16(np.around(circles))
            print(circles.shape[1], " centers detected!")
            cimg = drawCircles(cimg, circles)
            return cimg, circles
    else:
        print("Received None!")
        return None

def countOnSlice(CTcenters, T1centers):
    CTT1 = []
    CTT2 = []
    for CT in CTcenters[0]:
        for T1 in T1centers[0]:
            dist = math.sqrt((CT[0] - T1[0]) ** 2 + (CT[1] - T1[1]) ** 2)
            if dist < 10:
                CTT1.append(dist)
    return CTT1, CTT2


pathCT = input("Enter the absolute path of the CT image:\n")
imageCT = dcmread(pathCT)
pathMR = input("Enter the absolute path of the MR image:\n")
imageMR = dcmread(pathMR)
imagemodalityCT = imageCT.Modality
imagemodalityMR = imageMR.Modality
plt.imsave('imagetotalCT.png', imageCT.pixel_array)
plt.imsave('imagetotalMR.png', imageMR.pixel_array)
imgCT = cv2.imread('imagetotal.png', 0)
imgMR = cv2.imread('imagetotal.png', 0)

scale_percent = 100
if "CT" in imagemodalityCT:
    print("Received CT image!")
    scale_percent = 1000
    img = upscaleImage(imgCT, scale_percent)
if "MR" in imagemodalityMR:
    print("Received MR image!")
    scale_percent = 1500
    img = upscaleImage(imgMR, scale_percent)

maskedimgCT = maskROI(imgCT, imagemodalityCT)
maskedimgMR = maskROI(imgMR, imagemodalityMR)
detectionCT = detectCircles(maskedimgCT, imagemodalityCT)
detectionMR = detectCircles(maskedimgMR, imagemodalityMR)

if detectionCT is not None:
    CTcenters = detectionCT[0]
    if detectionMR is not None:
        T1centers = detectionMR[1]
    CTT1discrepancies = countOnSlice(CTcenters, T1centers)
    print(CTT1discrepancies[0])
    cimg = detectionCT[0]
    if cimg is not None:
        plt.imshow(cimg)
        plt.show()
        cv2.imwrite('detectedcircles.png', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


