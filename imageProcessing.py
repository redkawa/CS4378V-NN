import os
import skimage as skimage
from skimage import color, io
from skimage.transform import resize
import numpy as np
import random
from random import randint

def processSampleImages():
    # Process test high-fives (if necessary):

    path = './samples_hf'
    origPath = 'samples_hf/'
    savePath = 'samplesgray_hf/'
    endPath = './samplesgray_hf'

    os.makedirs(os.path.dirname('samplesgray_hf/'), exist_ok=True)

    num_files = len([f for f in os.listdir(endPath)
                     if os.path.isfile(os.path.join(endPath, f))])
    if num_files == 0:

        for pictureName in os.listdir(path):
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            convertedPicture = convertImageObject(importedPicture)
            putInGrayscaleFolder = savePath + pictureName
            skimage.io.imsave(putInGrayscaleFolder, convertedPicture)

    os.makedirs(os.path.dirname('samplesgray_nhf/'), exist_ok=True)

    path = './samples_nhf'
    origPath = 'samples_nhf/'
    savePath = 'samplesgray_nhf/'
    endPath = './samplesgray_nhf'
    num_files = len([f for f in os.listdir(endPath)
                     if os.path.isfile(os.path.join(endPath, f))])
    if num_files == 0:

        for pictureName in os.listdir(path):
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            convertedPicture = convertImageObject(importedPicture)
            putInGrayscaleFolder = savePath + pictureName
            skimage.io.imsave(putInGrayscaleFolder, convertedPicture)

    pictureHeight = 30
    pictureWidth = 30
    allSampleData = []

    endPath = './samplesgray_hf'
    origPath = 'samplesgray_hf/'
    for pictureName in os.listdir(endPath):
        if (pictureName != '.DS_Store'):
            data_set = [1.0000000000, 0.0000000000] #[1, 0] means the y set that indicates it IS a high five
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            for h in range(pictureHeight):
                for w in range(pictureWidth):
                    normalized = float(importedPicture[h][w])/ 255 # Normalize it
                    data_set.append(normalized)
            allSampleData.append(data_set)

    endPath = './samplesgray_nhf'
    origPath = 'samplesgray_nhf/'
    for pictureName in os.listdir(endPath):
        if (pictureName != '.DS_Store'):
            data_set = [0.0000000000, 1.0000000000] #[0, 1] means the y set that indicates it is NOT a high five
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            for h in range(pictureHeight):
                for w in range(pictureWidth):
                    normalized = float(importedPicture[h][w])/ 255 # Normalize it
                    data_set.append(normalized)
            allSampleData.append(data_set)

    sampleData = np.asarray(allSampleData)
    np.savetxt("sample_data.csv", sampleData, delimiter=",")

def processImages():

    # Process training high fives (if necessary):

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

    # Process training non-high fives (if necessary):

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

    # Create training non-high fives randomly (if necessary):

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

    # Process test high-fives (if necessary):

    path = './testhighfives'
    origPath = 'testhighfives/'
    savePath = 'testhighfivesgray/'
    endPath = './testhighfivesgray'
    num_files = len([f for f in os.listdir(endPath)
                     if os.path.isfile(os.path.join(endPath, f))])
    if num_files == 0:

        for pictureName in os.listdir(path):
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            convertedPicture = convertImageObject(importedPicture)
            putInGrayscaleFolder = savePath + pictureName
            skimage.io.imsave(putInGrayscaleFolder, convertedPicture)

    # Process test not high-fives (if necessary):

    path = './testnothighfives'
    origPath = 'testnothighfives/'
    savePath = 'testnothighfivesgray/'
    endPath = './testnothighfivesgray'
    num_files = len([f for f in os.listdir(endPath)
                     if os.path.isfile(os.path.join(endPath, f))])
    if num_files == 0:

        for pictureName in os.listdir(path):
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            convertedPicture = convertImageObject(importedPicture)
            putInGrayscaleFolder = savePath + pictureName
            skimage.io.imsave(putInGrayscaleFolder, convertedPicture)

    # Create training non-high fives randomly (if necessary):

    savePath = 'testnothighfivesrandomgray/'
    endPath = './testnothighfivesrandomgray'
    num_files = len([f for f in os.listdir(endPath)
                     if os.path.isfile(os.path.join(endPath, f))])
    if num_files == 0:

        numRandomPictures = 10
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
    # Take all the grayscaled images from highfivesgray, nothighfivesgray, and nothighfivesrandomgray,
    # Convert all the values in the matrixes from 0-255 to 0.0 to 1.0
    # And then put them all in the data directory

    pictureHeight = 30
    pictureWidth = 30
    allTrainingData = []
    allTestData = []

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
            allTrainingData.append(data_set)

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
            allTrainingData.append(data_set)

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
            allTrainingData.append(data_set)

    endPath = './testhighfivesgray'
    origPath = 'testhighfivesgray/'
    for pictureName in os.listdir(endPath):
        if (pictureName != '.DS_Store'):
            data_set = [1.0000000000, 0.0000000000] #[1, 0] means the y set that indicates it IS a high five
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            for h in range(pictureHeight):
                for w in range(pictureWidth):
                    normalized = float(importedPicture[h][w])/ 255 # Normalize it
                    data_set.append(normalized)
            allTestData.append(data_set)

    endPath = './testnothighfivesgray'
    origPath = 'testnothighfivesgray/'
    for pictureName in os.listdir(endPath):
        if (pictureName != '.DS_Store'):
            data_set = [0.0000000000, 1.0000000000] #[0, 1] means the y set that indicates it is NOT a high five
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            for h in range(pictureHeight):
                for w in range(pictureWidth):
                    normalized = float(importedPicture[h][w])/ 255 # Normalize it
                    data_set.append(normalized)
            allTestData.append(data_set)

    endPath = './testnothighfivesrandomgray'
    origPath = 'testnothighfivesrandomgray/'
    for pictureName in os.listdir(endPath):
        if (pictureName != '.DS_Store'):
            data_set = [0.0000000000, 1.0000000000] #[0, 1] means the y set that indicates it is NOT a high five
            pathToOrigFile = origPath + pictureName
            importedPicture = skimage.io.imread(pathToOrigFile)
            for h in range(pictureHeight):
                for w in range(pictureWidth):
                    normalized = float(importedPicture[h][w])/ 255 # Normalize it
                    data_set.append(normalized)
            allTestData.append(data_set)

    trainingData = np.asarray(allTrainingData)
    np.savetxt("training_data.csv", trainingData, delimiter=",")

    testData = np.asarray(allTestData)
    np.savetxt("test_data.csv", testData, delimiter=",")


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