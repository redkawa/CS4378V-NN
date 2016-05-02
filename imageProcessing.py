import os
import skimage as skimage
from skimage import color, io
from skimage.transform import resize


def processImages():
    path = './highfives'
    origPath = 'highfives/'
    savePath = 'highfivesgray/'
    num_files = len([f for f in os.listdir(path)
                     if os.path.isfile(os.path.join(path, f))])
    # num_files is the number of pictures in the directory /highfives

    for pictureName in os.listdir(path):
        pathToOrigFile = origPath + pictureName
        importedPicture = skimage.io.imread(pathToOrigFile)
        convertedPicture = convertImageObject(importedPicture)
        putInGrayscaleFolder = savePath + pictureName
        skimage.io.imsave(putInGrayscaleFolder, convertedPicture)

def printImageAsGrid(image):
    for array in image:
        for value in array:
            print(value, end="")
            print('   ', end="")
        print()

def convertImageObject(image):
    image2 = resize(image, (30, 30), mode='nearest') #mode='nearest' just means no border
    gray_img = color.rgb2gray(image2)
    return gray_img