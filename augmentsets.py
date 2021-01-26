import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
from glob import glob
import xml.etree.ElementTree as ET
import os
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from xml.dom import minidom

ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
filenames = [img for img in glob("spliceaug/*.jpg")]
annotations = [img for img in glob("spliceaug/*.xml")]

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.

processed_images = []

#img = cv2.cvtColor(cv2.imread("updated/beltvision01-web.jpg"), cv2.COLOR_BGR2RGB)

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
    
def hflip_image(img, bbs):

    images = np.array(
        [img for _ in range(1)],
        dtype=np.uint8
    )

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(1), # horizontally flip all images
            
        ],
        random_order=True
    )
    
    images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
    print(bbs_aug)
    return [images_aug, bbs_aug]

def vflip_image(img, bbs):

    images = np.array(
        [img for _ in range(1)],
        dtype=np.uint8
    )

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Flipud(1), # vertically flip all images
            
        ],
        random_order=True
    )
    
    images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
    print(bbs_aug)
    return [images_aug, bbs_aug]

def crop_image(img, bbs):

    images = np.array(
        [img for _ in range(1)],
        dtype=np.uint8
    )

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            #crop and pad the image
           iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )
            
        ],
        random_order=True
    )
    
    images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
    print(bbs_aug)
    return [images_aug, bbs_aug]

def affine_image(img, bbs):

    images = np.array(
        [img for _ in range(1)],
        dtype=np.uint8
    )

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            #crop and pad the image
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )
            
        ],
        random_order=True
    )
    
    images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
    print(bbs_aug)
    return [images_aug, bbs_aug]

def superpixel_image(img, bbs):

    images = np.array(
        [img for _ in range(1)],
        dtype=np.uint8
    )

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            #crop and pad the image
            iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))
            
        ],
        random_order=True
    )
    
    images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
    print(bbs_aug)
    return [images_aug, bbs_aug]

def blur_image(img, bbs):

    images = np.array(
        [img for _ in range(1)],
        dtype=np.uint8
    )

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            #crop and pad the image
            iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
            
        ],
        random_order=True
    )
    
    images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
    print(bbs_aug)
    return [images_aug, bbs_aug]

def augment_image(img, bbs):

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    images = np.array(
        [img for _ in range(1)],
        dtype=np.uint8
    )

    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(1), # horizontally flip 50% of all images
            iaa.Flipud(1), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-5, 5), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    #iaa.OneOf([
                    #    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    #    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    #    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    #]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        #iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.LinearContrast((0.5, 2.0))
                        )
                    ]),
                    iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    
    images_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
    print(bbs_aug)
    return [images_aug, bbs_aug]

def recoirdio_annotation_builder(VOC, aug_filename, bounding_array, aug_image):
    annotations = {}

    source = "source-ref"
    s3 = "s3://insights-use1-sdbxlz-beltvision-model-data/raw_annotations/"
    setname = "conveyor-condition-monitoring"
    filename = aug_filename

    #{"source-ref":"s3://conveyor-training-data/20190604_143222 01.jpg","conveyor-splice-labeling-with-damage":{"annotations":[{"class_id":0,"width":386,"top":404,"height":597,"left":1382},{"class_id":1,"width":92,"top":390,"height":93,"left":1666},{"class_id":1,"width":66,"top":554,"height":82,"left":1574}],"image_size":[{"width":1920,"depth":3,"height":1080}]},"conveyor-splice-labeling-with-damage-metadata":{"job-name":"labeling-job/conveyor-splice-labeling-with-damage","class-map":{"1":"damage","0":"splice"},"human-annotated":"yes","objects":[{"confidence":0.09},{"confidence":0.09},{"confidence":0.09}],"creation-date":"2019-09-22T15:05:41.732187","type":"groundtruth/object-detection"}}

    annotation_string = '{"' + source + '":"' + s3 + filename + '","' + setname + '":{"annotations":['
    objects_string = '"objects":['

    object_count = 0

    for objects in VOC:
        #get the class of the annotation
        name = objects.find('name').text
        anno_class = "0"

        if name == "splice":
            anno_class = "0"
        if name == "splice_vulcanized":
            anno_class = "1"
        if name == "damage":
            anno_class = "2"

        int(round(x))

        imgheight, imgwidth, channels = aug_image.shape

        xmin = bounding_array[object_count][0]
        ymin = bounding_array[object_count][1]
        xmax = bounding_array[object_count][2]
        ymax = bounding_array[object_count][3]

        width = max(0, int(round(xmax - xmin)))
        height = max(0, int(round(ymax - ymin)))
        top = max(0, int(round(ymin)))
        left = max(0, int(round(xmin)))

        if width and height > 0:

            annotation_string += '{"class_id":'+ str(anno_class) +',"width":' + str(width) + ',"top":' + str(top) + ',"height":' + str(height) + ',"left":' + str(left) + '}'
            
            if object_count + 1 < len(VOC):
                annotation_string +=','
                objects_string += '{"confidence":0.09},'
            else:
                annotation_string += '],'
                objects_string += '{"confidence":0.09}],'

            obects_string = ''
        
        object_count = object_count + 1
    
    annotation_string += '"image_size":[{"width":'+ str(imgwidth) +',"depth":'+ str(channels) + ',"height":'+ str(imgheight) + '}]},'
    annotation_string += '"'+ setname + '-metadata":{"job-name":"labeling-job/conveyor-splice-labeling-with-damage2","class-map":{"2":"damage","1":"splice_vulcanized","0":"splice"},"human-annotated":"yes",'
    annotation_string += objects_string
    annotation_string += '"creation-date":"2019-10-16T15:05:41.732187","type":"groundtruth/object-detection"}'
    annotation_string += '}'

    print(annotation_string)
    return annotation_string

def voc_annotation_builder(VOC, aug_filename, bounding_array,  aug_image):

    file_name = os.path.splitext(aug_filename)[0]
    imgheight, imgwidth, channels = aug_image.shape
   
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "voc_annotations"
    ET.SubElement(root, "filename").text = str(aug_filename)
    ET.SubElement(root, "path").text = "/home/ivan/ComputerVision/beltvision-training/captures" + str(aug_filename)
    
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "unkown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(imgwidth)
    ET.SubElement(size, "height").text = str(imgheight)
    ET.SubElement(size, "depth").text = str(channels)

    ET.SubElement(root, "segmented").text = str(0)
    
    object_count = 0

    for objects in VOC:

        xmin = bounding_array[object_count][0]
        ymin = bounding_array[object_count][1]
        xmax = bounding_array[object_count][2]
        ymax = bounding_array[object_count][3]

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = objects.find('name').text
        ET.SubElement(obj, "pose").text = objects.find('pose').text
        ET.SubElement(obj, "truncated").text = objects.find('truncated').text
        ET.SubElement(obj, "difficult").text = objects.find('difficult').text
        
        bndbox = ET.SubElement(obj, "bndbox")

        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

        object_count = object_count + 1

    pretty_root = ET.fromstring(prettify(root))
    tree = ET.ElementTree(pretty_root)
    tree.write("augmented/" + file_name + ".xml")

count = 0

annotations = []

#Append the augemented images to a list
for image in filenames:
    print(image)
    name = os.path.splitext(image)[0]
    print(name)
    img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    root = ET.parse(name +".xml").getroot()

    boxes = []
    VOC_root = root.findall('object')

    for objects in VOC_root:
        box = []
        box.append(int(objects.find('bndbox/xmin').text))
        boxes.append(BoundingBox(x1=int(objects.find('bndbox/xmin').text), x2=int(objects.find('bndbox/xmax').text), y1=int(objects.find('bndbox/ymin').text), y2=int(objects.find('bndbox/ymax').text)))
        
   
    bbs = BoundingBoxesOnImage(boxes, shape=img.shape)
    #print(bbs)
    #ia.imshow(bbs.draw_on_image(img, size=3))

    #aug_hflip = hflip_image(img, bbs)
    #aug_vflip = vflip_image(img, bbs)
    #aug_crop = crop_image(img,bbs)
    #aug_affine = affine_image(img,bbs)
    #aug_superpixel = superpixel_image(img,bbs)
    #aug_blur = blur_image(img,bbs)

    

    for x in range(1, 32):
        augmented = augment_image(img,bbs)
        bounding_array = augmented[1].to_xyxy_array()

        print(bounding_array)
        #ia.imshow(augmented[1].draw_on_image(augmented[0][0], size=3))
        print("Augementing...")
        cv2.imwrite("augmented/augment"+str(count)+".jpg", augmented[0][0])
        annotation = recoirdio_annotation_builder(VOC_root, "augment"+str(count)+".jpg", bounding_array, augmented[0][0])
        voc_annotation =  voc_annotation_builder(VOC_root, "augment"+str(count)+".jpg", bounding_array, augmented[0][0])
        

        annotations.append(annotation)
        count = count + 1

#finally we can write out our annotations
#annotation_file = open("output.manifest", "a")
#annotation_file.write("\n".join(annotations))
#annotation_file.close()

    #print(augemented[1])
    #ia.imshow(augemented[1].draw_on_image(augemented[0], size=3))
    #processed_images.append(augemented)
    
    
#Iterate through the list and create the augmented files
