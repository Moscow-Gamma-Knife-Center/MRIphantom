from pydicom import dcmread
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime

now = datetime.now()

#we multiply dx and dy by 1.01 for easier visual representation
def drawArrow(x1, y1, x2, y2):
    plt.arrow(x1, y1, 10*(x2 - x1), 10*(y2 - y1),
              head_width=0.1, length_includes_head=True)

def arrayFill(CTNames, T1Names, T2Names, CTXPoints, CTYPoints, T1XPoints, T1YPoints, T2XPoints, T2YPoints):
    T1NamesNew = [None] * len(CTNames)
    T2NamesNew = [None] * len(CTNames)
    T1XPointsNew = [None] * len(CTXPoints)
    T2XPointsNew = [None] * len(CTYPoints)
    T1YPointsNew = [None] * len(CTXPoints)
    T2YPointsNew = [None] * len(CTYPoints)
    counter = 0
    counternew = 0
    if len(T1Names) != 0:
        for i in CTNames:
            if (counter < len(T1Names)) and (i[:i.find('_')] != T1Names[counter][:T1Names[counter].find('_')]):
                T1NamesNew[counternew] = None
                T1XPointsNew[counternew] = None
                T1YPointsNew[counternew] = None
                counternew += 1
            elif counter < len(T1Names):
                T1NamesNew[counternew] = T1Names[counter]
                T1XPointsNew[counternew] = T1XPoints[counter]
                T1YPointsNew[counternew] = T1YPoints[counter]
                counternew += 1
                counter += 1
    else:
        print('Unfortunately, there are no T1 contours on this slice.')
    counter = 0
    counternew = 0
    if len(T2Names) != 0:
        for i in CTNames:
            if (counter < len(T2Names)) and (i[:i.find('_')] not in T2Names[counter]):
                T2NamesNew[counternew] = None
                T2XPointsNew[counternew] = None
                T2YPointsNew[counternew] = None
                counternew += 1
            elif counter < len(T2Names):
                T2NamesNew[counternew] = T2Names[counter]
                T2XPointsNew[counternew] = T2XPoints[counter]
                T2YPointsNew[counternew] = T2YPoints[counter]
                counternew += 1
                counter += 1
    else:
        print('Unfortunately, there are no T2 contours on this slice.')
    return T1NamesNew, T2NamesNew, T1XPointsNew, T1YPointsNew, T2XPointsNew, T2YPointsNew

def plotContours(contour_dataset, image, zcoord, savingPath):
    img_ID = image.SOPInstanceUID
    img = image
    img_arr = img.pixel_array
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])
    origin_x, origin_y, _ = img.ImagePositionPatient
    rows = []
    cols = []
    # массивы для хранения номеров контуров
    CTNumbers = []
    T1Numbers = []
    T2Numbers = []
    CTNames = []
    T1Names = []
    T2Names = []
    # массивы контуров
    CTX = []
    CTY = []
    T1X = []
    T1Y = []
    T2X = []
    T2Y = []
    # буфер
    CTXT = []
    CTYT = []
    T1XT = []
    T1YT = []
    T2XT = []
    T2YT = []
    # массивы для центров контуров
    CTXPoints = []
    T1XPoints = []
    T2XPoints = []
    CTYPoints = []
    T1YPoints = []
    T2YPoints = []
    # инициализация графиков
    fig, ax = plt.subplots(1, figsize=(60, 50))
    # заполнение массивов номерами контуров, чтобы произвести дальнейшее распределение контуров по своим массивам, согласно их номераам
    for ObservationT in contour_dataset.RTROIObservationsSequence:
        if 'CT' in ObservationT.ROIObservationLabel:
            CTNumbers.append(int(ObservationT.ReferencedROINumber))
        if 'T1' in ObservationT.ROIObservationLabel:
            T1Numbers.append(int(ObservationT.ReferencedROINumber))
        if 'T2' in ObservationT.ROIObservationLabel:
            T2Numbers.append(int(ObservationT.ReferencedROINumber))
    # обработка данных
    for ROIContourSequenceNumber in contour_dataset.ROIContourSequence:
        for t in range(0, len(ROIContourSequenceNumber.ContourSequence)):
            contour_coordinates = ROIContourSequenceNumber.ContourSequence[t].ContourData
            # x, y, z координаты контура в миллиметрах
            x0 = contour_coordinates[len(contour_coordinates) - 3]
            y0 = contour_coordinates[len(contour_coordinates) - 2]
            z0 = contour_coordinates[len(contour_coordinates) - 1]
            coordinates = []

            if (contour_coordinates[2] == zcoord):
                tempry = -1
                # преобразование
                for i in range(0, len(contour_coordinates), 3):
                    x = contour_coordinates[i]
                    y = contour_coordinates[i + 1]
                    z = contour_coordinates[i + 2]
                    l = math.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0))
                    l = math.ceil(l * 2) + 1
                    for k in range(1, l + 1):
                        coordinates.append([(x - x0) * k / l + x0, (y - y0) * k / l + y0, (z - z0) * k / l + z0])
                    x0 = x
                    y0 = y
                    z0 = z
                    # заполнение массивов контуров CT T1 T2
                    if ROIContourSequenceNumber.ReferencedROINumber in CTNumbers:
                        CTX.append((x - origin_x) / x_spacing)
                        CTY.append((y - origin_y) / y_spacing)
                        CTXT.append((x - origin_x) / x_spacing)
                        CTYT.append((y - origin_y) / y_spacing)
                        # этот блок ниже отвечает за прорисовку контура
                        if tempry != ROIContourSequenceNumber.ReferencedROINumber:
                            CTNames.append(contour_dataset.RTROIObservationsSequence[
                                               ROIContourSequenceNumber.ReferencedROINumber - 1].ROIObservationLabel)
                            xt = CTXT[-1]
                            yt = CTYT[-1]
                            CTXT.pop()  # ПОСЛЕДНИЙ ЭЛЕМЕНТ НАДО ПЕРЕНЕСТИ В НАЧАЛО СЛЕДУЮЩЕГО КОНТУРА
                            CTYT.pop()
                            # на ноль делить нельзя, поэтому откидываем эти варианты (это не выкинет точки, все на месте)
                            if len(CTXT) != 0:
                                CTXPoints.append(sum(CTXT) / len(CTXT))
                                CTYPoints.append(sum(CTYT) / len(CTYT))
                            else:
                                CTXPoints.append(None)
                                CTYPoints.append(None)
                            ax.plot(CTXT, CTYT, color='g', linestyle='-')
                            CTXT.clear()
                            CTYT.clear()
                            CTXT.append(xt)
                            CTYT.append(yt)
                    if ROIContourSequenceNumber.ReferencedROINumber in T1Numbers:
                        T1X.append((x - origin_x) / x_spacing)
                        T1Y.append((y - origin_y) / y_spacing)
                        T1XT.append((x - origin_x) / x_spacing)
                        T1YT.append((y - origin_y) / y_spacing)
                        # этот блок ниже отвечает за прорисовку контура
                        if tempry != ROIContourSequenceNumber.ReferencedROINumber:
                            T1Names.append(contour_dataset.RTROIObservationsSequence[
                                               ROIContourSequenceNumber.ReferencedROINumber - 1].ROIObservationLabel)
                            xt = T1XT[-1]
                            yt = T1YT[-1]
                            T1XT.pop()
                            T1YT.pop()
                            # на ноль делить нельзя, поэтому откидываем эти варианты (это не выкинет точки, все на месте)
                            if len(T1XT) != 0:
                                T1XPoints.append(sum(T1XT) / len(T1XT))
                                T1YPoints.append(sum(T1YT) / len(T1YT))
                            else:
                                T1XPoints.append(None)
                                T1YPoints.append(None)
                            ax.plot(T1XT, T1YT, color='r', linestyle='-')
                            T1XT.clear()
                            T1YT.clear()
                            T1XT.append(xt)
                            T1YT.append(yt)
                    if ROIContourSequenceNumber.ReferencedROINumber in T2Numbers:
                        T2X.append((x - origin_x) / x_spacing)
                        T2Y.append((y - origin_y) / y_spacing)
                        T2XT.append((x - origin_x) / x_spacing)
                        T2YT.append((y - origin_y) / y_spacing)
                        # этот блок ниже отвечает за прорисовку контура
                        if tempry != ROIContourSequenceNumber.ReferencedROINumber:
                            T2Names.append(contour_dataset.RTROIObservationsSequence[
                                               ROIContourSequenceNumber.ReferencedROINumber - 1].ROIObservationLabel)
                            xt = T2XT[-1]
                            yt = T2YT[-1]
                            T2XT.pop()
                            T2YT.pop()
                            # на ноль делить нельзя, поэтому откидываем эти варианты (это не выкинет точки, все на месте)
                            if len(T2XT) != 0:
                                T2XPoints.append(sum(T2XT) / len(T2XT))
                                T2YPoints.append(sum(T2YT) / len(T2YT))
                            else:
                                T2XPoints.append(None)
                                T2YPoints.append(None)
                            ax.plot(T2XT, T2YT, color='b', linestyle='-')
                            T2XT.clear()
                            T2YT.clear()
                            T2XT.append(xt)
                            T2YT.append(yt)
                    tempry = ROIContourSequenceNumber.ReferencedROINumber
                tempry = ROIContourSequenceNumber.ReferencedROINumber
            pixel_coordinates = [(np.round((y - origin_y) / y_spacing), np.round((x - origin_x) / x_spacing)) for
                                 x, y, _ in coordinates]
            for i, j in list(set(pixel_coordinates)):
                rows.append(i)
                cols.append(j)
            contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8,
                                     shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

    # графики для центров контуров
    plt.scatter(CTXPoints, CTYPoints, color='g', marker='.')
    plt.scatter(T1XPoints, T1YPoints, color='r', marker='.')
    plt.scatter(T2XPoints, T2YPoints, color='b', marker='.')

    CTNames = CTNames[4:]
    T1Names = T1Names[4:]
    T2Names = T2Names[4:]

    array = arrayFill(CTNames, T1Names, T2Names, CTXPoints, CTYPoints, T1XPoints, T1YPoints, T2XPoints, T2YPoints)
    print('CTNames: ', CTNames)
    print('T1Names: ', array[0])
    print('T2Names: ', array[1])

    T1XPoints = array[2].copy()
    T1YPoints = array[3].copy()
    T2XPoints = array[4].copy()
    T2YPoints = array[5].copy()

    print('CT CENTERS: ', len(CTXPoints), '  ', CTXPoints[:])
    print('CT CENTERS: ', len(CTYPoints), '  ', CTYPoints[:])

    print('T1 CENTERS: ', len(T1XPoints), '  ', T1XPoints[:])
    print('T1 CENTERS: ', len(T1YPoints), '  ', T1YPoints[:])

    print('T2 CENTERS: ', len(T2XPoints), '  ', T2XPoints[:])
    print('T2 CENTERS: ', len(T2YPoints), '  ', T2YPoints[:])



    # draw arrows
    for i in range(4, len(T1XPoints)):
        if (T1XPoints[i] is not None) and (math.sqrt((CTXPoints[i] - T1XPoints[i]) ** 2 + (CTYPoints[i] - T1YPoints[i]) ** 2) < 10):
            drawArrow(CTXPoints[i], CTYPoints[i], T1XPoints[i], T1YPoints[i])
    for i in range(4, len(T2XPoints)):
        if (T2XPoints[i] is not None) and (math.sqrt((CTXPoints[i] - T2XPoints[i]) ** 2 + (CTYPoints[i] - T2YPoints[i]) ** 2) < 10):
            drawArrow(CTXPoints[i], CTYPoints[i], T2XPoints[i], T2YPoints[i])

    fig = plt.gcf()
    fig.savefig(f'{savingPath}/slice{zcoord}.png')
    print(f'saved slice {zcoord}')
    plt.close('all')

    return img_arr, contour_arr, img_ID, CTX, CTY, T1X, T1Y, T2X, T2Y

def findPossibleZ(contour_dataset):
    print("Possible slices: ")
    possibleZ = []
    for q in range(0, len(contour_dataset.ROIContourSequence)):
        for p in range(0, len(contour_dataset.ROIContourSequence[q].ContourSequence)):
            contour_coordinates = contour_dataset.ROIContourSequence[q].ContourSequence[p].ContourData
            for i in range(0, len(contour_coordinates), 3):
                z = contour_coordinates[i + 2]
                possibleZ.append(z)
    possibleZ = list(set(possibleZ))
    possibleZ.sort()
    for x in possibleZ: print(x)
    return possibleZ

if __name__ == "__main__":
    currentPath = os.getcwd()
    try:
        os.mkdir(currentPath + '/pictures')
    except:
        print("The directory /pictures is already created")

    #create a folder for current time
    os.mkdir(currentPath + '/pictures/' + now.strftime("%d.%m.%Y.%H.%M.%S"))
    savingPath = currentPath + '/pictures/' + now.strftime("%d.%m.%Y.%H.%M.%S")

    path = 'DICOM/'
    filename = 'RTSS.dcm'
    image = 'IMG0000000100.dcm'
    ds = dcmread(path + filename)
    image = dcmread(path + image)

    zinterest = findPossibleZ(ds)
    print("Starting saving images...")
    for z in zinterest:
        plotContours(ds, image, z, savingPath)

