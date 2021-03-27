import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg


class IndexTracker:
    def __init__(self, ax, avalaibleImages, numbers):
        self.ax = ax
        ax.set_title('Use scroll wheel to navigate slices')

        self.slices = len(avalaibleImages)
        self.ind = 0
        self.z = numbers[0]

        img = mpimg.imread(avalaibleImages[0])
        self.im = ax.imshow(img)
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
            self.z = numbers[self.ind + 1]
        else:
            self.ind = (self.ind - 1) % self.slices
            self.z = numbers[self.ind - 1]
        self.update()

    def update(self):
        img = mpimg.imread(avalaibleImages[self.ind])
        self.im = ax.imshow(img)
        self.ax.set_ylabel('slice %s' % self.z)
        self.im.axes.figure.canvas.draw()

def findFolders(path):
    avalaibleDirectories = []
    print("Avalaible data:")
    for f in os.scandir(path):
        if f.is_dir():
            print(f.name)
            avalaibleDirectories.append(f.name)
    return avalaibleDirectories

def findImages(path):
    avalaibleImages = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            avalaibleImages.append(os.path.join(path, file))
    return avalaibleImages

def getNumbers(Images):
    resultImages = []
    for Image in Images:
        resultImages.append(Image[Image.find("slice") + 5:Image.find(".png")])
    return resultImages

picturesPath = os.getcwd() + '\\pictures\\'
findFolders(picturesPath)
foldername = input("Enter the name of the folder:\n")
avalaibleImages = findImages(picturesPath + str(foldername))
numbers = getNumbers(avalaibleImages)

fig, ax = plt.subplots(1, 1)

tracker = IndexTracker(ax, avalaibleImages, numbers)

fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
plt.show()