from imutils import paths
import mxnet as mx
import numpy as np
import argparse
import imutils
import cv2
import shutil
import operator
import re
import os
import sys, getopt
import boto3
import psycopg2
import collections
import statistics
import time

from pathlib import Path
from glob import glob

from random import shuffle

path = "dataset/"
trainingdir = "training/"
validationdir = "validation/"

imagePaths = glob(path + "*.jpg")
imagePaths.extend(glob(path + "*.png"))

shuffle(imagePaths)

voctotal = len(imagePaths)

train, validate = np.split(imagePaths, [int(len(imagePaths)*0.8)])

print("Training Sets:" + str(len(train)))
print("Validation Sets:" + str(len(validate)))

#Delete the training directory if it already exists
if os.path.isdir(trainingdir) == True:
    shutil.rmtree(trainingdir)

#Check if training directory exists, if no create
Path(trainingdir).mkdir(parents=True, exist_ok=True)

for image in train:
    filename = os.path.basename(image)
    filebase = os.path.splitext(filename)[0]
    xmlfile = filebase + ".xml"
    xmlpath = path + xmlfile
    shutil.copy(image, "training/")
    shutil.copy(xmlpath, "training/")


#Delete the validation directory if it already exists
if os.path.isdir(validationdir) == True:
    shutil.rmtree(validationdir)

#Check if validation directory exists, if no create
Path(validationdir).mkdir(parents=True, exist_ok=True)


for image in validate:
    filename = os.path.basename(image)
    filebase = os.path.splitext(filename)[0]
    xmlfile = filebase + ".xml"
    xmlpath = path + xmlfile
    shutil.copy(image, "validation/")
    shutil.copy(xmlpath, "validation/")


print("Done!")