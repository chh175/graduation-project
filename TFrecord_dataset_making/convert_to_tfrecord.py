from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import random

import tensorflow as tf

#原始数据目录
tf.app.flags.DEFINE_string('dataset_dir', 'lfw', '')
#输出目录
tf.app.flags.DEFINE_string('output_dir', 'lfw_tfrecord', '')
#转换成tfrecord时，是否打乱数据集顺序
tf.app.flags.DEFINE_bool('shuffle', False, '')


tf.app.flags.DEFINE_string('label_name', 'labels.txt', '')
#输出文件的名字
tf.app.flags.DEFINE_string('output_file_name', 'align_lfw', '')
#转换成训练集的比例
tf.app.flags.DEFINE_float('train_valid_split', 1.0, '')
#总共要分成多少个文件
tf.app.flags.DEFINE_integer('num_shards', 5, 'number of shards to split')



FLAGS = tf.app.flags.FLAGS




def write_label_file(labels_to_class_names, dataset_dir,
                     filename=FLAGS.label_name):
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


def int64_feature(values):    
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):    
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):    
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      FLAGS.output_file_name, split_name, shard_id, FLAGS.num_shards)
  return os.path.join(dataset_dir, output_filename)


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  
  directories = []
  class_names = []
  for filename in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)



def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  _NUM_SHARDS = FLAGS.num_shards
  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


  
def main(_):
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)
      
    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    
    # Divide into train and test:
    random.seed(666)
    if FLAGS.shuffle:
        random.shuffle(photo_filenames)
    
    num_train = int(len(photo_filenames) * FLAGS.train_valid_split)
    
    training_filenames = photo_filenames[:num_train]
    validation_filenames = photo_filenames[num_train:]
    
    
    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                       FLAGS.output_dir)
    
    if FLAGS.train_valid_split < 1:
        _convert_dataset('validation', validation_filenames, class_names_to_ids,
                       FLAGS.output_dir)
    
    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, FLAGS.output_dir)

    
    print('\nFinished converting the dataset!')
    print('num_train', len(training_filenames))
    print('num_validation', len(validation_filenames))
    print('num_class', len(class_names))
    
    dataset_information = os.path.join(FLAGS.output_dir, 'dataset_information.txt')    
    with open(dataset_information,'w') as f:
        f.write('num_train = %d\n' % len(training_filenames))
        f.write('num_validation = %d\n' % len(validation_filenames))
        f.write('num_class = %d\n' % len(class_names))
        


if __name__ == '__main__':
    tf.app.run()
  
  