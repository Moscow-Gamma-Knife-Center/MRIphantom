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
            # retval, img = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)
            ROIcircle = np.uint16(np.around(ROIcircle))
            mask = np.zeros_like(img)
            for i in ROIcircle[0, :]:
                # draw the outer circle
                mask = cv2.circle(mask, (i[0], i[1]), (i[2] - 30), (255, 255, 0), -1)
            maskedimg = cv2.bitwise_and(img, mask)
            return maskedimg, ROIcircle
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
            return maskedimg, ROIcircle
    else:
        print("Can't find ROI on this image...!")
        return None

def drawCircles(img, circles):
    # draw circles, centers and their number
    print("drawing circles....")
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

def detectCircles(img, modality, ROIcenter):
    if img is not None:
        if "MR" in modality:
            print("detecting circles on MRI...")
            cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # we use cimg for final render
            distant = 210
            p1 = 40
            p2 = 2
            minR = 20
            maxR = 28

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask , (cx - int((1.2 / 5) * r), cy - r), (cx + int((1.2 / 5) * r), cy - int((4.5/5) * r)), (255, 255, 0), -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles1 = cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                       param1=p1, param2=p2, minRadius=minR, maxRadius=maxR)
            circles1 = np.uint16(np.around(circles1))
            circles1[0] = circles1[0][circles1[0][:, 0].argsort()]
            circles = circles1

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((3 / 5) * r), cy - int((3.9 / 5) * r)), (cx + int((3 / 5) * r), cy - int((3.55 / 5) * r)), (255, 255, 0), -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles2 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                       param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles2 = np.uint16(np.around(circles2))
            circles2[0] = circles2[0][circles2[0][:, 0].argsort()]
            circles = np.append(circles, circles2, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4 / 5) * r), cy - int((3 / 5) * r)), (cx + int((4 / 5) * r), cy - int((2.65 / 5) * r)), (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles3 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles3 = np.uint16(np.around(circles3))
            circles3[0] = circles3[0][circles3[0][:, 0].argsort()]
            circles = np.append(circles, circles3, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4 / 5) * r), cy - int((2.1 / 5) * r)), (cx + int((4 / 5) * r), cy - int((1.65 / 5) * r)), (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles4 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles4 = np.uint16(np.around(circles4))
            circles4[0] = circles4[0][circles4[0][:, 0].argsort()]
            circles = np.append(circles, circles4, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4.8 / 5) * r), cy - int((1.15 / 5) * r)), (cx + int((4.8 / 5) * r), cy - int((0.6 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles5 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles5 = np.uint16(np.around(circles5))
            circles5[0] = circles5[0][circles5[0][:, 0].argsort()]
            circles = np.append(circles, circles5, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4.8 / 5) * r), cy - int((0.25 / 5) * r)), (cx + int((4.8 / 5) * r), cy + int((0.3 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles6 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles6 = np.uint16(np.around(circles6))
            circles6[0] = circles6[0][circles6[0][:, 0].argsort()]
            circles = np.append(circles, circles6, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4.8 / 5) * r), cy + int((0.7 / 5) * r)), (cx + int((4.8 / 5) * r), cy + int((1.2 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles7 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles7 = np.uint16(np.around(circles7))
            circles7[0] = circles7[0][circles7[0][:, 0].argsort()]
            circles = np.append(circles, circles7, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4 / 5) * r), cy + int((1.6 / 5) * r)), (cx + int((4 / 5) * r), cy + int((2.1 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles8 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles8 = np.uint16(np.around(circles8))
            circles8[0] = circles8[0][circles8[0][:, 0].argsort()]
            circles = np.append(circles, circles8, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4 / 5) * r), cy + int((2.55 / 5) * r)), (cx + int((4 / 5) * r), cy + int((3.05 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles9 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles9 = np.uint16(np.around(circles9))
            circles9[0] = circles9[0][circles9[0][:, 0].argsort()]
            circles = np.append(circles, circles9, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((3 / 5) * r), cy + int((3.5 / 5) * r)), (cx + int((3 / 5) * r), cy + int((3.95 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles10 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles10 = np.uint16(np.around(circles10))
            circles10[0] = circles10[0][circles10[0][:, 0].argsort()]
            circles = np.append(circles, circles10, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((1.2 / 5) * r), cy + int((4.45 / 5) * r)), (cx + int((1.2 / 5) * r), cy + int((5.1 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles11 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles11 = np.uint16(np.around(circles11))
            circles11[0] = circles11[0][circles11[0][:, 0].argsort()]
            circles = np.append(circles, circles11, axis=1)

            cimg = drawCircles(cimg, circles)
            plt.imshow(cimg)
            plt.show()
            return cimg, circles
        if "CT" in modality:
            print("detecting circles on CT...")
            cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            distant = 225
            p1 = 20
            p2 = 1
            minR = 20
            maxR = 32

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((1.2 / 5) * r), cy - r),
                                     (cx + int((1.2 / 5) * r), cy - int((4.5 / 5) * r)), (255, 255, 0), -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles1 = cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                        param1=p1, param2=p2, minRadius=minR, maxRadius=maxR)
            circles1 = np.uint16(np.around(circles1))
            circles1[0] = circles1[0][circles1[0][:, 0].argsort()]
            circles = circles1

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((3 / 5) * r), cy - int((3.9 / 5) * r)),
                                     (cx + int((3 / 5) * r), cy - int((3.55 / 5) * r)), (255, 255, 0), -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles2 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles2 = np.uint16(np.around(circles2))
            circles2[0] = circles2[0][circles2[0][:, 0].argsort()]
            circles = np.append(circles, circles2, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4 / 5) * r), cy - int((3 / 5) * r)),
                                     (cx + int((4 / 5) * r), cy - int((2.65 / 5) * r)), (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles3 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles3 = np.uint16(np.around(circles3))
            circles3[0] = circles3[0][circles3[0][:, 0].argsort()]
            circles = np.append(circles, circles3, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4 / 5) * r), cy - int((2.1 / 5) * r)),
                                     (cx + int((4 / 5) * r), cy - int((1.65 / 5) * r)), (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles4 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles4 = np.uint16(np.around(circles4))
            circles4[0] = circles4[0][circles4[0][:, 0].argsort()]
            circles = np.append(circles, circles4, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4.8 / 5) * r), cy - int((1.15 / 5) * r)),
                                     (cx + int((4.8 / 5) * r), cy - int((0.6 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles5 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles5 = np.uint16(np.around(circles5))
            circles5[0] = circles5[0][circles5[0][:, 0].argsort()]
            circles = np.append(circles, circles5, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4.8 / 5) * r), cy - int((0.25 / 5) * r)),
                                     (cx + int((4.8 / 5) * r), cy + int((0.3 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles6 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles6 = np.uint16(np.around(circles6))
            circles6[0] = circles6[0][circles6[0][:, 0].argsort()]
            circles = np.append(circles, circles6, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4.8 / 5) * r), cy + int((0.7 / 5) * r)),
                                     (cx + int((4.8 / 5) * r), cy + int((1.2 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles7 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles7 = np.uint16(np.around(circles7))
            circles7[0] = circles7[0][circles7[0][:, 0].argsort()]
            circles = np.append(circles, circles7, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4 / 5) * r), cy + int((1.6 / 5) * r)),
                                     (cx + int((4 / 5) * r), cy + int((2.1 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles8 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles8 = np.uint16(np.around(circles8))
            circles8[0] = circles8[0][circles8[0][:, 0].argsort()]
            circles = np.append(circles, circles8, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((4 / 5) * r), cy + int((2.55 / 5) * r)),
                                     (cx + int((4 / 5) * r), cy + int((3.05 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles9 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                         param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles9 = np.uint16(np.around(circles9))
            circles9[0] = circles9[0][circles9[0][:, 0].argsort()]
            circles = np.append(circles, circles9, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((3 / 5) * r), cy + int((3.5 / 5) * r)),
                                     (cx + int((3 / 5) * r), cy + int((3.95 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles10 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                          param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles10 = np.uint16(np.around(circles10))
            circles10[0] = circles10[0][circles10[0][:, 0].argsort()]
            circles = np.append(circles, circles10, axis=1)

            img_copy = img
            ROIcenter = np.uint16(np.around(ROIcenter))
            mask = np.zeros_like(img_copy)
            for i in ROIcenter[0, :]:
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                mask = cv2.rectangle(mask, (cx - int((1.2 / 5) * r), cy + int((4.45 / 5) * r)),
                                     (cx + int((1.2 / 5) * r), cy + int((5.1 / 5) * r)),
                                     (255, 255, 0),
                                     -1)
            maskedimg = cv2.bitwise_and(img_copy, mask)
            # detect circles
            circles11 = (cv2.HoughCircles(maskedimg, cv2.HOUGH_GRADIENT, 1, distant,
                                          param1=p1, param2=p2, minRadius=minR, maxRadius=maxR))
            circles11 = np.uint16(np.around(circles11))
            circles11[0] = circles11[0][circles11[0][:, 0].argsort()]
            circles = np.append(circles, circles11, axis=1)

            cimg = drawCircles(cimg, circles)
            plt.imshow(cimg)
            plt.show()
            return cimg, circles
    else:
        print("Received None!")
        return None

def countOnSlice(CTcenters, T1centers, scale_percent_CT, scale_percent_MR):
    CTT1 = []
    CTT2 = []
    for CT in CTcenters[0]:
        for T1 in T1centers[0]:
            x1 = CT[0] / scale_percent_CT * 100
            x2 = T1[0] / scale_percent_MR * 100
            y1 = CT[1] / scale_percent_CT * 100
            y2 = T1[1] / scale_percent_MR * 100
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            # if dist < 100:
            CTT1.append(dist)
    return CTT1, CTT2


# pathCT = input("Enter the absolute path of the CT image:\n")
pathCT = "C:\\Users\\bzavo\\Documents\\MRIphantom\\MRIphantom\\MR_Phantom_scans\\DICOM_CT_contour\\IMG0000000100.dcm"
imageCT = dcmread(pathCT)
# pathMR = input("Enter the absolute path of the MR image:\n")
pathMR = "C:\\Users\\bzavo\\Documents\\MRIphantom\\MRIphantom\\MR_Phantom_scans\\DICOM_MR_contour\\IMG0000000100.dcm"
imageMR = dcmread(pathMR)
imagemodalityCT = imageCT.Modality
imagemodalityMR = imageMR.Modality
plt.imsave('imagetotalCT.png', imageCT.pixel_array)
plt.imsave('imagetotalMR.png', imageMR.pixel_array)
imgCT = cv2.imread('imagetotalCT.png', 0)
imgMR = cv2.imread('imagetotalMR.png', 0)

scale_percent_CT = 1000
scale_percent_MR = 1500
if "CT" in imagemodalityCT:
    print("Received CT image!")
    imgCT = upscaleImage(imgCT, scale_percent_CT)
if "MR" in imagemodalityMR:
    print("Received MR image!")
    imgMR = upscaleImage(imgMR, scale_percent_MR)

maskedimgCT = maskROI(imgCT, imagemodalityCT)
maskedimgMR = maskROI(imgMR, imagemodalityMR)
detectionCT = detectCircles(maskedimgCT[0], imagemodalityCT, maskedimgCT[1])
detectionMR = detectCircles(maskedimgMR[0], imagemodalityMR, maskedimgMR[1])

if detectionCT is not None:
    CTcenters = detectionCT[0]
    if detectionMR is not None:
        T1centers = detectionMR[1]
    CTT1discrepancies = countOnSlice(CTcenters, T1centers, scale_percent_CT, scale_percent_MR)
    print(min(CTT1discrepancies[0]))
    cimg = detectionCT[0]
    if cimg is not None:
        plt.imshow(cimg)
        plt.show()
        cv2.imwrite('detectedcircles.png', cimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


