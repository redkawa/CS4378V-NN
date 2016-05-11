from scipy import ndimage as ndi
import matplotlib
import os.path
import skimage as skimage
from skimage import color, io
from skimage.transform import resize
import random
import numpy as np
import tensorflow as tf
import imageProcessing
import time
import gatherResults

start = time.time()
imageProcessing.setup()
gatherResults.gather_statistics()
end = time.time()
print("Time it took (in seconds): " + str(end - start))
#imageProcessing.printImageAsGrid(skimage.io.imread('highfivesgray/21-human-front.jpg'))