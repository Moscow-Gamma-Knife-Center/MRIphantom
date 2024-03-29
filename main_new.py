from pydicom import dcmread
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pandas as pd
import scipy.stats as stats

def countdistances(contour_dataset, image, zcoord, distancesCTT1, distancesCTT2):
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

    # обсчет точек по тройкам CT-T1 CT-T2 T1-T2
    # начинается не от 0, а от 3, потому что отброшены первые три значения
    for i in range(3, len(T1XPoints)):
        distancesCTT1.append(math.sqrt((CTXPoints[i] - T1XPoints[i]) ** 2 + (CTYPoints[i] - T1YPoints[i]) ** 2))
    for i in range(3, len(T2XPoints)):
        distancesCTT2.append(math.sqrt((CTXPoints[i] - T2XPoints[i]) ** 2 + (CTYPoints[i] - T2YPoints[i]) ** 2))

    return distancesCTT1, distancesCTT2

def funSelection(contour_dataset, image, zcoord, selection05, selection1):
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
    for q in selection05:
        plt.scatter(CTXPoints[q-3], CTYPoints[q-3], color='k', marker='o', alpha=.3, s=1000)
    #fig = plt.gcf()
    plt.show()
    # plt.draw()
    # fig.savefig('contours.png', dpi=300)

    return 1

def fun(contour_dataset, image, zcoord):
    print("Начинаю чтение файла...")
    img_ID = image.SOPInstanceUID
    img = image
    img_arr = img.pixel_array
    x_spacing, y_spacing = float(img.PixelSpacing[0]), float(img.PixelSpacing[1])
    origin_x, origin_y, _ = img.ImagePositionPatient
    rows = []
    cols = []
    #массивы для хранения номеров контуров
    CTNumbers = []
    T1Numbers = []
    T2Numbers = []
    #массивы контуров
    CTX = []
    CTY = []
    T1X = []
    T1Y = []
    T2X = []
    T2Y = []
    #буфер
    CTXT = []
    CTYT = []
    T1XT = []
    T1YT = []
    T2XT = []
    T2YT = []
    #массивы для центров контуров
    CTXPoints = []
    T1XPoints = []
    T2XPoints = []
    CTYPoints = []
    T1YPoints = []
    T2YPoints = []
    #инициализация графиков
    fig, ax = plt.subplots(1, figsize=(60, 50))
    #заполнение массивов номерами контуров, чтобы произвести дальнейшее распределение контуров по своим массивам, согласно их номераам
    for ObservationT in contour_dataset.RTROIObservationsSequence:
        if 'CT' in ObservationT.ROIObservationLabel:
            CTNumbers.append(int(ObservationT.ReferencedROINumber))
        if 'T1' in ObservationT.ROIObservationLabel:
            T1Numbers.append(int(ObservationT.ReferencedROINumber))
        if 'T2' in ObservationT.ROIObservationLabel:
            T2Numbers.append(int(ObservationT.ReferencedROINumber))
#обработка данных
    print("Начинаю обработку извлечённых данных...")
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
                #преобразование
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
                    #заполнение массивов контуров CT T1 T2
                    if ROIContourSequenceNumber.ReferencedROINumber in CTNumbers:
                        CTX.append((x - origin_x) / x_spacing)
                        CTY.append((y - origin_y) / y_spacing)
                        CTXT.append((x - origin_x) / x_spacing)
                        CTYT.append((y - origin_y) / y_spacing)
                        #этот блок ниже отвечает за прорисовку контура
                        if tempry != ROIContourSequenceNumber.ReferencedROINumber:
                            xt = CTXT[-1]
                            yt = CTYT[-1]
                            CTXT.pop() #ПОСЛЕДНИЙ ЭЛЕМЕНТ НАДО ПЕРЕНЕСТИ В НАЧАЛО СЛЕДУЮЩЕГО КОНТУРА
                            CTYT.pop()
                            # на ноль делить нельзя, поэтому откидываем эти варианты (это не выкинет точки, все на месте)
                            if len(CTXT) != 0:
                                CTXPoints.append(sum(CTXT) / len(CTXT))
                                CTYPoints.append(sum(CTYT) / len(CTYT))
                            # if len(CTXT) == 0:
                            #     CTXPoints.append(sum(CTX) / len(CTX))
                            # if len(CTYT) == 0:
                            #     CTYPoints.append(sum(CTY) / len(CTY))
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
                            #на ноль делить нельзя, поэтому откидываем эти варианты (это не выкинет точки, все на месте)
                            if len(T1XT) != 0 :
                                T1XPoints.append(sum(T1XT) / len(T1XT))
                                T1YPoints.append(sum(T1YT) / len(T1YT))
                            # if len(T1XT) == 0:
                            #     T1XPoints.append(sum(T1X) / len(T1X))
                            # if len(CTYT) == 0:
                            #     T1YPoints.append(sum(T1Y) / len(T1Y))
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
                            # if len(T2XT) == 0:
                            #     T2XPoints.append(sum(T2X) / len(T2X))
                            # if len(CTYT) == 0:
                            #     T2YPoints.append(sum(T2Y) / len(T2Y))
                            ax.plot(T2XT, T2YT, color='b', linestyle='-')
                            T2XT.clear()
                            T2YT.clear()
                            T2XT.append(xt)
                            T2YT.append(yt)
                    tempry = ROIContourSequenceNumber.ReferencedROINumber
                tempry = ROIContourSequenceNumber.ReferencedROINumber
            pixel_coordinates = [(np.round((y - origin_y) / y_spacing), np.round((x - origin_x) / x_spacing)) for x, y, _ in coordinates]
            for i, j in list(set(pixel_coordinates)):
                rows.append(i)
                cols.append(j)
            contour_arr = csc_matrix((np.ones_like(rows), (rows, cols)), dtype=np.int8, shape=(img_arr.shape[0], img_arr.shape[1])).toarray()

    #графики для центров контуров
    print("Создаю изображение...")
    plt.scatter(CTXPoints, CTYPoints, color='g', marker='.')
    plt.scatter(T1XPoints, T1YPoints, color='r', marker='.')
    plt.scatter(T2XPoints, T2YPoints, color='b', marker='.')
    plt.show()

    plt.figure(1)
    # # обсчет точек по тройкам CT-T1 CT-T2 T1-T2
    # distancesCTT1 = []
    # distancesCTT2 = []
    #
    # for i in range(3, len(T1XPoints)):
    #     distancesCTT1.append(math.sqrt((CTXPoints[i] - T1XPoints[i]) ** 2 + (CTYPoints[i] - T1YPoints[i]) ** 2))
    # for i in range(3, len(T2XPoints)):
    #     distancesCTT2.append(math.sqrt((CTXPoints[i] - T2XPoints[i]) ** 2 + (CTYPoints[i] - T2YPoints[i]) ** 2))
    # print(distancesCTT1)
    # print(len(distancesCTT1), "/ 88")
    # print(distancesCTT2)
    # print(len(distancesCTT2), "/ 88")
    # #число точек, чье отклонение больше 0.5 и 1 мм
    # selection05 = [] #массив номеров точек, которые надо выделить 0.5 mm
    # selection1 = []  # массив номеров точек, которые надо выделить 1 mm
    # more05T1 = 0
    # more1T1 = 0
    # for i in range(0, len(distancesCTT1)):
    #     if distancesCTT1[i] > 0.5:
    #         more05T1 += 1
    #         selection05.append(i)
    #     if distancesCTT1[i] > 1:
    #         more1T1 += 1
    #         selection1.append(i)
    # print("number of points where CT T1 > 0.5 is  ", more05T1, "/", len(distancesCTT1))
    # print("number of points where CT T1 > 1 is  ", more1T1, "/", len(distancesCTT1))
    #
    # if len(distancesCTT1) != 0:
    #     meandistanceCTT1 = np.mean(distancesCTT1)
    #     print("mean distance CT T1 = ", round(meandistanceCTT1, 5), " mm")
    #     print("max distance CT T1 = ", round(max(distancesCTT1), 5), " mm")
    #     print("min distance CT T1 = ", round(min(distancesCTT1), 5), " mm")
    # # число точек, чье отклонение больше 0.5 и 1 мм
    # more05T2 = 0
    # more1T2 = 0
    # for i in range(0, len(distancesCTT2)):
    #     if distancesCTT2[i] > 0.5:
    #         more05T2 += 1
    #         selection05.append(i)
    #     if distancesCTT2[i] > 1:
    #         more1T2 += 1
    #         selection1.append(i)
    # print("number of points where CT T2 > 0.5 is  ", more05T2, "/", len(distancesCTT2))
    # print("number of points where CT T2 > 1 is  ", more1T2, "/", len(distancesCTT2))
    # if len(distancesCTT2) != 0:
    #     meandistanceCTT2 = np.mean(distancesCTT2)
    #     print("mean distance CT T2 = ", round(meandistanceCTT2, 5), " mm")
    #     print("max distance CT T2 = ", round(max(distancesCTT2), 5), " mm")
    #     print("min distance CT T2 = ", round(min(distancesCTT2), 5), " mm")
    # selection05 = list(set(selection05))
    # selection1 = list(set(selection1))

    return img_arr, contour_arr, img_ID, CTX, CTY, T1X, T1Y, T2X, T2Y #, selection05, selection1

#найти все доступные координаты z контуров
def zcoords(contour_dataset):
    print("Ищу возможные варианты...")
    possibleZ = []
    for q in range(0, len(contour_dataset.ROIContourSequence)):
        for p in range(0, len(contour_dataset.ROIContourSequence[q].ContourSequence)):
            contour_coordinates = contour_dataset.ROIContourSequence[q].ContourSequence[p].ContourData
            for i in range(0, len(contour_coordinates), 3):
                z = contour_coordinates[i + 2]
                possibleZ.append(z)
    possibleZ = list(set(possibleZ))
    possibleZ.sort()
    return possibleZ


if __name__ == "__main__":
    #путь до папки
    path = 'C:\\Users\\bzavo\\PycharmProjects\\MainProject\\DICOM\\'
    filename = 'RTSS.dcm'
    image = 'IMG0000000100.dcm'
    ds = dcmread(path + filename)
    image = dcmread(path + image)
    #все данные есть при z=-14

    possibleZ = zcoords(ds)
    print("Доступные срезы:")
    for x in possibleZ:
        print(x)
    zcurrent = float(input("Введите координату z: "))
    fun(ds, image, zcurrent)

    # t = fun(ds, image, zcoord)

    # funSelection(ds, image, zcoord, t[9], t[10])