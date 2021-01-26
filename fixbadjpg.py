import imghdr
import cv2
import os
import glob

for folder in ['training', 'validation']:
    image_path = os.path.join(os.getcwd(), (folder))
    print(image_path)
    print(len(glob.glob(image_path + '/*.jpg')))
for file in glob.glob(image_path + '/*.jpg'):
    image = cv2.imread(file)
    file_type = imghdr.what(file)
    if file_type != 'jpeg':
        print(file + " - invalid - " + str(file_type))
    # cv2.imwrite(file, image)