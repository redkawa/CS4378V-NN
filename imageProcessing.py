import os
import skimage as skimage
from skimage import color, io
from skimage.transform import resize
import numpy as np
import random
from random import randint


def processImages():

    #Process high fives (if necessary):

    path = './highfives'
    origPath = 'highfives/'
    savePath = 'highfivesgray/'
    endPath = './highfivesgray'
    num_files = len([f for f in os.listdir(endPath)
                     if os.path.isfile(os.path.join(endPath, f))])
    if num_files == 0:

        for pictureName in os.listdir(path):
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            convertedPicture = convertImageObject(importedPicture)
            putInGrayscaleFolder = savePath + pictureName
            skimage.io.imsave(putInGrayscaleFolder, convertedPicture)

    #Process non-high fives (if necessary):

    path = './nothighfives'
    origPath = 'nothighfives/'
    savePath = 'nothighfivesgray/'
    endPath = './nothighfivesgray'
    num_files = len([f for f in os.listdir(endPath)
                     if os.path.isfile(os.path.join(endPath, f))])
    if num_files == 0:

        for pictureName in os.listdir(path):
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            convertedPicture = convertImageObject(importedPicture)
            putInGrayscaleFolder = savePath + pictureName
            skimage.io.imsave(putInGrayscaleFolder, convertedPicture)

    #Create non-high fives randomly (if necessary):

    savePath = 'nothighfivesrandomgray/'
    endPath = './nothighfivesrandomgray'
    num_files = len([f for f in os.listdir(endPath)
                     if os.path.isfile(os.path.join(endPath, f))])
    if num_files == 0:

        numRandomPictures = 50
        randomPictureHeight = 30
        randomPictureWidth = 30
        for i in range(numRandomPictures):
            #Create 50 30x30 px pictures with random values from 0 to 255
            pictureMatrix = []
            for h in range(randomPictureHeight):
                widthVector = []
                for w in range(randomPictureWidth):
                    widthVector.append(random.randint(0, 255))
                pictureMatrix.append(widthVector)

            pictureName = 'randomPicture_' + str(i + 1) + '.jpg'
            putInGrayscaleFolder = savePath + pictureName
            skimage.io.imsave(putInGrayscaleFolder, pictureMatrix)

def processData():
    #Take all the grayscaled images from highfivesgray, nothighfivesgray, and nothighfivesrandomgray,
    #Convert all the values in the matrixes from 0-255 to 0.0 to 1.0
    #And then put them all in the data directory

    pictureHeight = 30
    pictureWidth = 30
    allData = []

    endPath = './highfivesgray'
    origPath = 'highfivesgray/'
    for pictureName in os.listdir(endPath):
        if (pictureName != '.DS_Store'):
            data_set = [1.0000000000, 0.0000000000] #[1, 0] means the y set that indicates it IS a high five
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            for h in range(pictureHeight):
                for w in range(pictureWidth):
                    normalized = float(importedPicture[h][w])/ 255 # Normalize it
                    data_set.append(normalized)
            allData.append(data_set)

    endPath = './nothighfivesgray'
    origPath = 'nothighfivesgray/'
    for pictureName in os.listdir(endPath):
        if (pictureName != '.DS_Store'):
            data_set = [0.0000000000, 1.0000000000] #[0, 1] means the y set that indicates it is NOT a high five
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            for h in range(pictureHeight):
                for w in range(pictureWidth):
                    normalized = float(importedPicture[h][w])/ 255 # Normalize it
                    data_set.append(normalized)
            allData.append(data_set)

    endPath = './nothighfivesrandomgray'
    origPath = 'nothighfivesrandomgray/'
    for pictureName in os.listdir(endPath):
        if (pictureName != '.DS_Store'):
            data_set = [0.0000000000, 1.0000000000] #[0, 1] means the y set that indicates it is NOT a high five
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            for h in range(pictureHeight):
                for w in range(pictureWidth):
                    normalized = float(importedPicture[h][w])/ 255 # Normalize it
                    data_set.append(normalized)
            allData.append(data_set)

    data = np.asarray(allData)
    np.savetxt("data.csv", data, delimiter=",")


def printImageAsGrid(image):
    for array in image:
        for value in array:
            print(value, end="")
            print('   ', end="")
        print()
    print('Height: ' + str(len(image)) + '   Width: ' + str(len(image[0])))
    print('0 is 100% black, 255 is 100% white')

def convertImageObject(image):
    image2 = resize(image, (30, 30), mode='nearest') #mode='nearest' just means no border
    gray_img = color.rgb2gray(image2)
    return gray_img

def setup():
    # converts all images to grayscale, shrinks them, moves them into their corresponding folders,
    # then takes all converted images and converts them into matrices,
    # then adds all those matrices together into one big matrix,
    # and saves that matrix into a csv file called data.csv
    processImages()
    processData()