import os
import sys
import random

import numpy as np
import tensorflow as tf
from PIL import Image

import xml.etree.ElementTree as ET

from io import BytesIO

#from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
#from datasets.pascalvoc_common import VOC_LABELS

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'validation/'
DIRECTORY_IMAGES = 'validation/'
OUTPUT = 'tfrecord/'
NAME = 'validation'

classes = ['splice', 'splice_vulcanized', 'damage']

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

LABELS_FILENAME = 'labels.txt'

VOC_LABELS = {
    'splice': (0, 'splice'),
    'splice_vulcanized': (1, 'splice_vulcanized'),
    'damage': (2, 'damage')
}

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convertToJpeg(im):
    with BytesIO() as f:
        im.save(f, format='JPEG')
        return f.getvalue()

def _process_image(directory, name):
    
    """Process a image and annotation file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    file_only = name + '.jpg'
    filename = directory + name + '.jpg'
    image_data = tf.gfile.FastGFile(filename, 'rb').read()

    #print(filename)
    # Read the XML annotation file.
    filename = directory + name + '.xml'
    
    tree = ET.parse(filename)
    print(filename)
    root = tree.getroot()

    print("Root file:" + str(root))

    # Image shape.
    size = root.find('size')
    shape = [int(size.find('height').text),
             int(size.find('width').text),
             int(size.find('depth').text)]
    # Find annotations.
    bboxes = []
    labels = []
    labels_text = []
    difficult = []
    truncated = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        labels.append(int(VOC_LABELS[label][0]))
        labels_text.append(label.encode('ascii'))

        if obj.find('difficult'):
            difficult.append(int(obj.find('difficult').text))
        else:
            difficult.append(0)
        if obj.find('truncated'):
            truncated.append(int(obj.find('truncated').text))
        else:
            truncated.append(0)

        bbox = obj.find('bndbox')
        bboxes.append((float(bbox.find('ymin').text) / shape[0],
                       float(bbox.find('xmin').text) / shape[1],
                       float(bbox.find('ymax').text) / shape[0],
                       float(bbox.find('xmax').text) / shape[1]
                       ))
    return image_data, shape, bboxes, labels, labels_text, difficult, truncated, file_only

def _normalize_bounding_box(xmins, ymins, xmaxs, ymaxs):

    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []

    for xmin in xmins:
        if xmin < 0:
            xmin = 0
        if xmin > 1:
            xmin = 1
        xmin_list.append(xmin)

    for ymin in ymins:
        if ymin < 0:
            ymin = 0
        if ymin > 1:
            ymin = 1
        ymin_list.append(ymin)    
    
    for xmax in xmaxs:
        if xmax > 1:
            xmax = 1
        if xmax < 0:
            xmax = 0
        xmax_list.append(xmax)  

    for ymax in ymaxs:
        if ymax > 1:
            ymax = 1
        if ymax < 0:
            ymax = 0
        ymax_list.append(ymax) 

    return xmin_list, ymin_list, xmax_list, ymax_list

def _convert_to_example(image_data, labels, labels_text, bboxes, shape, difficult, truncated, file_only):
    

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned

    #check the bbox values and correct any issues

    xmin, ymin, xmax, ymax = _normalize_bounding_box(xmin,ymin,xmax,ymax)

    image_format = b'JPEG'
   
    example = tf.train.Example(
                        features= tf.train.Features(
                            feature={
                                'image/height': int64_feature(shape[0]),
                                'image/width': int64_feature(shape[1]),
                                'image/filename': bytes_feature(file_only.encode('utf-8')),
                                'image/source_id': bytes_feature(file_only.encode('utf-8')),
                                'image/format': bytes_feature(image_format),
                                'image/encoded': bytes_feature(image_data),
                                'image/object/bbox/xmin': float_list_feature(xmin),
                                'image/object/bbox/xmax': float_list_feature(xmax),
                                'image/object/bbox/ymin': float_list_feature(ymin),
                                'image/object/bbox/ymax': float_list_feature(ymax),
                                'image/object/class/text': bytes_list_feature(labels_text),
                                'image/object/class/label': int64_list_feature(labels),
                            }
                        )
                    )

    return example

def _add_to_tfrecord(dataset_dir, name, tfrecord_writer):
    """Loads data from image and annotations files and add them to a TFRecord.

    Args:
      dataset_dir: Dataset directory;
      name: Image name to add to the TFRecord;
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    #print("Directory:" + dataset_dir)
    image_data, shape, bboxes, labels, labels_text, difficult, truncated, file_only = \
        _process_image(dataset_dir, name)
    example = _convert_to_example(image_data, labels, labels_text,
                                  bboxes, shape, difficult, truncated, file_only)
    tfrecord_writer.write(example.SerializeToString())

def _get_output_filename(output_dir, name, idx):
    return OUTPUT + NAME + '.record'

def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.

    Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))

def generate_label_map(labels, dataset_dir):
        with open(dataset_dir + 'label_map.pbtxt', 'w') as label_map_file:
            for index, label in enumerate(labels):
                label_map_file.write('item {\n')
                label_map_file.write(' id: ' + str(index + 1) + '\n')
                label_map_file.write(" name: '" + label + "'\n")
                label_map_file.write('}\n\n')

def run(dataset_dir, output_dir, name='voc_train', shuffling=False):

    #print(dataset_dir)
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      output_dir: Output directory.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # Dataset filenames, and shuffling.
    #path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
    path = os.path.join(dataset_dir)

   
    filenames = sorted(os.listdir(path))
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(filenames)

    #print(filenames)

    # Process dataset files.
    i = 0
    fidx = 0
    while i < len(filenames):
        # Open new TFRecord file.
        tf_filename = _get_output_filename(output_dir, name, fidx)
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            j = 0
            while i < len(filenames):
                sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                sys.stdout.flush()
                
                filename = filenames[i]
                img_name = filename[:-4]
                #print("File:" + img_name)
                _add_to_tfrecord(dataset_dir, img_name, tfrecord_writer)
                i += 1
                j += 1
            fidx += 1

# Finally, write the labels file:

#labels_to_class_names = dict(zip(range(len(classes)), classes))
#write_label_file(labels_to_class_names, 'labels/')
#generate_label_map(classes, 'labels/')

#print('\nFinished converting the Pascal VOC dataset!')

run(DIRECTORY_IMAGES, OUTPUT)
