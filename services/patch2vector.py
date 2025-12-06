import cv2 as cv
import numpy as np

def patch_to_vector(patch, size=(56,46)):
    gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
    resized = cv.resize(gray, size)
    return resized.flatten()


