from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import keras
import cv2
import numpy as np

def iris_recognition(img):
    process = Processing(img)
    processed_img = process.get_iris()
    extraction = Extraction(img)
    features = extraction.get_features()
    classification = Classfification(img)
    label = classification.get_result_label()
    return label

# Load an image
img = cv2.imread('iris.jpg')

# Perform iris recognition
label = iris_recognition(img)

print(label)