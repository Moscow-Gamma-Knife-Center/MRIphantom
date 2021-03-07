from pydicom import dcmread
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

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
    return T1NamesNew, T2NamesNew, T1XPointsNew, T1YPointsNew, T2XPointsNew, T2YPointsNew


def plotContours(contour_dataset, image, zcoord):
    print("Начинаю чтение файла...")
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
    # fig, ax = plt.subplots(1, figsize=(60, 50))
    # заполнение массивов номерами контуров, чтобы произвести дальнейшее распределение контуров по своим массивам, согласно их номераам
    for ObservationT in contour_dataset.RTROIObservationsSequence:
        if 'CT' in ObservationT.ROIObservationLabel:
            CTNumbers.append(int(ObservationT.ReferencedROINumber))
        if 'T1' in ObservationT.ROIObservationLabel:
            T1Numbers.append(int(ObservationT.ReferencedROINumber))
        if 'T2' in ObservationT.ROIObservationLabel:
            T2Numbers.append(int(ObservationT.ReferencedROINumber))
    # обработка данных
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
                            if (len(CTXT) != 0) or (len(CTYT) != 0):
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
                            if (len(T1XT) != 0) or (len(T1YT) != 0):
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
                            if (len(T2XT) != 0) or (len(T2YT) != 0):
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

    return img_arr, contour_arr, img_ID, CTX, CTY, T1X, T1Y, T2X, T2Y, CTXPoints, CTYPoints, T1XPoints, T1YPoints, T2XPoints, T2YPoints,


if __name__ == "__main__":
    path = 'DICOM/'  # path of the folder with the files
    filename = 'RTSS.dcm'  # the name of file
    image = 'IMG0000000100.dcm'  # corresponding image
    ds = dcmread(path + filename)  # get the structures data
    image = dcmread(path + image)  # get the image data

    zcurrent = float(input("Введите координату z: "))
    # centerIndex = int(input("Введите индекс центров: "))
    plotdata = plotContours(ds, image, zcurrent)

    # fig, ax = plt.subplots(1, figsize=(60, 50))
    #
    # # plt.plot(plotdata[3], plotdata[4], color='g', linestyle='-')
    # # plt.plot(plotdata[5], plotdata[6], color='r', linestyle='-')
    # # plt.plot(plotdata[7], plotdata[8], color='b', linestyle='-')
    #
    # plt.scatter(plotdata[9][centerIndex], plotdata[10][centerIndex], color='g', linestyle='-')
    # plt.scatter(plotdata[11][centerIndex], plotdata[12][centerIndex], color='r', linestyle='-')
    # plt.scatter(plotdata[13][centerIndex], plotdata[14][centerIndex], color='b', linestyle='-')
    #
    # plt.show()
