from pydicom import dcmread
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import math
import pandas as pd

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

def countDistances(contour_dataset, image, zcoord, distancesCTT1, distancesCTT2):
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
                            T2XT.clear()
                            T2YT.clear()
                            T2XT.append(xt)
                            T2YT.append(yt)
                    tempry = ROIContourSequenceNumber.ReferencedROINumber
            pixel_coordinates = [(np.round((y - origin_y) / y_spacing), np.round((x - origin_x) / x_spacing)) for
                                 x, y, _ in coordinates]
            for i, j in list(set(pixel_coordinates)):
                rows.append(i)
                cols.append(j)
            contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8,
                                     shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

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

    print('CT CENTERS: ', len(CTXPoints), '  ', CTXPoints[4:])
    print('CT CENTERS: ', len(CTYPoints), '  ', CTYPoints[4:])

    print('T1 CENTERS: ', len(T1XPoints), '  ', T1XPoints[4:])
    print('T1 CENTERS: ', len(T1YPoints), '  ', T1YPoints[4:])

    print('T2 CENTERS: ', len(T2XPoints), '  ', T2XPoints[4:])
    print('T2 CENTERS: ', len(T2YPoints), '  ', T2YPoints[4:])

    # обсчет точек по тройкам CT-T1 CT-T2
    for i in range(4, len(T1XPoints)):
        if (CTXPoints[i] != None) and (CTYPoints[i] != None) and (T1XPoints[i] != None) and (T1YPoints[i] != None):
            dist = math.sqrt((CTXPoints[i] - T1XPoints[i]) ** 2 + (CTYPoints[i] - T1YPoints[i]) ** 2)
            if dist < 10:
                distancesCTT1.append(dist)
    for i in range(4, len(T2XPoints)):
        if (CTXPoints[i] != None) and (CTYPoints[i] != None) and (T2XPoints[i] != None) and (T2YPoints[i] != None):
            dist = math.sqrt((CTXPoints[i] - T2XPoints[i]) ** 2 + (CTYPoints[i] - T2YPoints[i]) ** 2)
            if dist < 10:
                distancesCTT2.append(dist)
    print('CT T1 DISTANCES: ', distancesCTT1)
    print('CT T2 DISTANCES: ', distancesCTT2)

    return distancesCTT1, distancesCTT2

def selectPoints(contour_dataset, image, zcoord, selection05, selection1):
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
    fig1, ax1 = plt.subplots(1, figsize=(60, 50))
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
                            xt = CTXT[-1]
                            yt = CTYT[-1]
                            CTXT.pop()  # ПОСЛЕДНИЙ ЭЛЕМЕНТ НАДО ПЕРЕНЕСТИ В НАЧАЛО СЛЕДУЮЩЕГО КОНТУРА
                            CTYT.pop()
                            # на ноль делить нельзя, поэтому откидываем эти варианты (это не выкинет точки, все на месте)
                            if len(CTXT) != 0:
                                CTXPoints.append(sum(CTXT) / len(CTXT))
                                CTYPoints.append(sum(CTYT) / len(CTYT))
                            # if len(CTXT) == 0:
                            #     CTXPoints.append(sum(CTX) / len(CTX))
                            # if len(CTYT) == 0:
                            #     CTYPoints.append(sum(CTY) / len(CTY))
                            ax1.plot(CTXT, CTYT, color='g', linestyle='-')
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
                            xt = T1XT[-1]
                            yt = T1YT[-1]
                            T1XT.pop()
                            T1YT.pop()
                            # на ноль делить нельзя, поэтому откидываем эти варианты (это не выкинет точки, все на месте)
                            if len(T1XT) != 0:
                                T1XPoints.append(sum(T1XT) / len(T1XT))
                                T1YPoints.append(sum(T1YT) / len(T1YT))
                            # if len(T1XT) == 0:
                            #     T1XPoints.append(sum(T1X) / len(T1X))
                            # if len(CTYT) == 0:
                            #     T1YPoints.append(sum(T1Y) / len(T1Y))
                            ax1.plot(T1XT, T1YT, color='r', linestyle='-')
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
                            xt = T2XT[-1]
                            yt = T2YT[-1]
                            T2XT.pop()
                            T2YT.pop()
                            # на ноль делить нельзя, поэтому откидываем эти варианты (это не выкинет точки, все на месте)
                            if len(T2XT) != 0:
                                T2XPoints.append(sum(T2XT) / len(T2XT))
                                T2YPoints.append(sum(T2YT) / len(T2YT))
                            # if len(T2XT) == 0:
                            #     T2XPoints.append(sum(T2X) / len(T2X))
                            # if len(CTYT) == 0:
                            #     T2YPoints.append(sum(T2Y) / len(T2Y))
                            ax1.plot(T2XT, T2YT, color='b', linestyle='-')
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
    for q in selection1:
        plt.scatter(CTXPoints[q - 3], CTYPoints[q - 3], color='k', marker='o', alpha=.3, s=1000)
    plt.show()
    return 1

def plotContours(contour_dataset, image, zcoord):
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
    # fig = plt.gcf()
    plt.show()
    # plt.draw()
    # fig.savefig('contours.png', dpi=300)
    plt.figure(1)
    # обсчет точек по тройкам CT-T1 CT-T2 T1-T2
    distancesCTT1 = []
    distancesCTT2 = []
    # ручная обработка первых трех выбивающихся точек (не очень сработало :с )
    # xa = CTXPoints[0]
    # ya = CTYPoints[0]
    # xb = CTXPoints[1]
    # yb = CTYPoints[1]
    #
    # CTXPoints[0] = CTXPoints[2]
    # CTYPoints[0] = CTYPoints[2]
    # CTXPoints[2] = xa
    # CTYPoints[2] = ya
    # CTXPoints[1] = CTXPoints[0]
    # CTYPoints[1] = CTYPoints[0]
    # CTXPoints[0] = xb
    # CTYPoints[0] = yb
    # начинается не от 0, а от 3, потому что отброшены первые три значения
    for i in range(3, len(T1XPoints)):
        distancesCTT1.append(math.sqrt((CTXPoints[i] - T1XPoints[i]) ** 2 + (CTYPoints[i] - T1YPoints[i]) ** 2))
    for i in range(3, len(T2XPoints)):
        distancesCTT2.append(math.sqrt((CTXPoints[i] - T2XPoints[i]) ** 2 + (CTYPoints[i] - T2YPoints[i]) ** 2))
    print(distancesCTT1)
    print(len(distancesCTT1), "/ 88")
    print(distancesCTT2)
    print(len(distancesCTT2), "/ 88")
    # число точек, чье отклонение больше 0.5 и 1 мм
    selection05 = []  # массив номеров точек, которые надо выделить 0.5 mm
    selection1 = []  # массив номеров точек, которые надо выделить 1 mm
    more05T1 = 0
    more1T1 = 0
    for i in range(0, len(distancesCTT1)):
        if distancesCTT1[i] > 0.5:
            more05T1 += 1
            selection05.append(i)
        if distancesCTT1[i] > 1:
            more1T1 += 1
            selection1.append(i)
    print("number of points where CT T1 > 0.5 is  ", more05T1, "/", len(distancesCTT1))
    print("number of points where CT T1 > 1 is  ", more1T1, "/", len(distancesCTT1))

    if len(distancesCTT1) != 0:
        meandistanceCTT1 = np.mean(distancesCTT1)
        print("mean distance CT T1 = ", round(meandistanceCTT1, 5), " mm")
        print("max distance CT T1 = ", round(max(distancesCTT1), 5), " mm")
        print("min distance CT T1 = ", round(min(distancesCTT1), 5), " mm")
    # число точек, чье отклонение больше 0.5 и 1 мм
    more05T2 = 0
    more1T2 = 0
    for i in range(0, len(distancesCTT2)):
        if distancesCTT2[i] > 0.5:
            more05T2 += 1
            selection05.append(i)
        if distancesCTT2[i] > 1:
            more1T2 += 1
            selection1.append(i)
    print("number of points where CT T2 > 0.5 is  ", more05T2, "/", len(distancesCTT2))
    print("number of points where CT T2 > 1 is  ", more1T2, "/", len(distancesCTT2))
    if len(distancesCTT2) != 0:
        meandistanceCTT2 = np.mean(distancesCTT2)
        print("mean distance CT T2 = ", round(meandistanceCTT2, 5), " mm")
        print("max distance CT T2 = ", round(max(distancesCTT2), 5), " mm")
        print("min distance CT T2 = ", round(min(distancesCTT2), 5), " mm")
    selection05 = list(set(selection05))
    selection1 = list(set(selection1))

    # графики для отклонений T1 и T2 от СТ
    if ((len(distancesCTT1) != 0) and (len(distancesCTT2) != 0)):
        d = dict(T1=np.array(distancesCTT1), T2=np.array(distancesCTT2))
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
        df = df.plot.kde()
    # df = pd.DataFrame({
    #     'T1': distancesCTT1,
    #     'T2': distancesCTT2,
    # })
    plt.show()

    return img_arr, contour_arr, img_ID, CTX, CTY, T1X, T1Y, T2X, T2Y, selection05, selection1

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
    path = 'DICOM/'
    filename = 'RTSS.dcm'
    image = 'IMG0000000100.dcm'
    ds = dcmread(path + filename)
    image = dcmread(path + image)

    zinterest = findPossibleZ(ds)

    distancesCTT1 = []
    distancesCTT2 = []
    for zslice in zinterest:
        print("Z = ", zslice, " mm is processing")
        countDistances(ds, image, zslice, distancesCTT1, distancesCTT2)

    # число точек, чье отклонение больше 0.5 и 1 мм
    selection05 = []  # массив номеров точек, которые надо выделить 0.5 mm
    selection1 = []  # массив номеров точек, которые надо выделить 1 mm
    more05T1 = 0
    more1T1 = 0
    for i in range(0, len(distancesCTT1)):
        if distancesCTT1[i] > 0.5:
            more05T1 += 1
            selection05.append(i)
        if distancesCTT1[i] > 1:
            more1T1 += 1
            selection1.append(i)
    print("number of points where CT T1 > 0.5 is  ", more05T1, "/", len(distancesCTT1))
    print("number of points where CT T1 > 1 is  ", more1T1, "/", len(distancesCTT1))

    if len(distancesCTT1) != 0:
        meandistanceCTT1 = np.mean(distancesCTT1)
        print("mean distance CT T1 = ", round(meandistanceCTT1, 5), " mm")
        stdT1 = np.std(distancesCTT1)
        print("standard deviation CT T1 = ", round(stdT1, 5), " mm")
        print("max distance CT T1 = ", round(max(distancesCTT1), 5), " mm")
        print("min distance CT T1 = ", round(min(distancesCTT1), 5), " mm")
    # число точек, чье отклонение больше 0.5 и 1 мм
    more05T2 = 0
    more1T2 = 0
    for i in range(0, len(distancesCTT2)):
        if distancesCTT2[i] > 0.5:
            more05T2 += 1
            selection05.append(i)
        if distancesCTT2[i] > 1:
            more1T2 += 1
            selection1.append(i)
    print("number of points where CT T2 > 0.5 is  ", more05T2, "/", len(distancesCTT2))
    print("number of points where CT T2 > 1 is  ", more1T2, "/", len(distancesCTT2))
    if len(distancesCTT2) != 0:
        meandistanceCTT2 = np.mean(distancesCTT2)
        print("mean distance CT T2 = ", round(meandistanceCTT2, 5), " mm")
        stdT2 = np.std(distancesCTT2)
        print("standard deviation CT T2 = ", round(stdT2, 5), " mm")
        print("max distance CT T2 = ", round(max(distancesCTT2), 5), " mm")
        print("min distance CT T2 = ", round(min(distancesCTT2), 5), " mm")
    selection05 = list(set(selection05))
    selection1 = list(set(selection1))

    if ((len(distancesCTT1) != 0) and (len(distancesCTT2) != 0)):
        d = dict(T1=np.array(distancesCTT1), T2=np.array(distancesCTT2))
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
        df = df.plot.kde()
    plt.show()

    t = plotContours(ds, image, 6)
    selectPoints(ds, image, 6, t[9], t[10])
