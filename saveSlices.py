from pydicom import dcmread
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import math
import os
from datetime import datetime

now = datetime.now()

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

