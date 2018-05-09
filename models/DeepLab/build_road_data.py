import os.path
import argparse
import io
import os
import sys
import PIL

from os.path import join
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='./road/data',
                    help='Path to the directory containing the preprocessed data.')

parser.add_argument('--output_dir', type=str, default='./road/tfrecord',
                    help='Path to the directory to create TFRecords outputs.')

_NUM_SHARDS = 4

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def dict_to_tf_example(image_path, label_path):
  """ Convert image and label to tf.Example proto.

  Args:
    image_path: Path to a single SATELLITE image.
    label_path: Path to its corresponding label.

  Returns:
    example: The converted tf.Example.

  """

  # read image
  with tf.gfile.FastGFile(image_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  with tf.gfile.FastGFile(label_path, 'rb') as fid:
    encoded_label = fid.read()

  width, height = image.size


  example = tf.train.Example(features=tf.train.Features(feature={
    'image/filename': bytes_feature(image_path.split('_')[0].split('/')[1].encode('utf8')),
    'image/channels': int64_feature(3),
    'image/height': int64_feature(height),
    'image/width': int64_feature(width),
    'image/encoded': bytes_feature(encoded_jpg),
    'image/format': bytes_feature('jpg'.encode('utf8')),
    'image/segmentation/class/encoded': bytes_feature(encoded_label),
    'image/segmentation/class/format': bytes_feature('png'.encode('utf8')),
  }))

  return example


def create_tf_record(output_filename, dir, examples):
  """Creates a TFRecord file from examples.
  Args:
    output_filename: Path to where output file is saved.
    dir: Directory where images and masks are stored.
    examples: Examples to parse and save to tf record, [[_sat.jpg,_mask.png],].
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 500 == 0:
      tf.logging.info('On image %d of %d', idx, len(examples))
    image_path = join(dir, example[0])
    label_path = join(dir, example[1])

    if not os.path.exists(image_path):
      tf.logging.warning('Could not find %s, ignoring example.', image_path)
      continue
    elif not os.path.exists(label_path):
      tf.logging.warning('Could not find %s, ignoring example.', label_path)
      continue

    try:
      tf_example = dict_to_tf_example(image_path, label_path)
      writer.write(tf_example.SerializeToString())
    except ValueError:
      tf.logging.warning('Invalid example: %s, ignoring.', example)

  writer.close()


def main(unused_argv):
  tf.logging.info("Reading from original dataset...")

  train_dir = join(FLAGS.data_dir, 'train')
  valid_dir = join(FLAGS.data_dir, 'valid')
  train_data_list = join(FLAGS.data_dir, 'train.txt')
  valid_data_list = join(FLAGS.data_dir, 'valid.txt')

  if not os.path.isdir(train_dir):
    raise ValueError("Missing image directory.")
  if not os.path.isdir(valid_dir):
    raise ValueError("Missing Augmentation label directory.")

  with tf.gfile.GFile(train_data_list) as fid:
    lines = fid.readlines()
  train_examples = [line.strip().split(' ') for line in lines]

  with tf.gfile.GFile(valid_data_list) as fid:
    lines = fid.readlines()
  val_examples = [line.strip().split(' ') for line in lines]

  if not os.path.isdir(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  train_output_dir = join(FLAGS.output_dir, 'train-00000-of-00001.record')
  val_output_dir = join(FLAGS.output_dir, 'val-00000-of-00001.record')

  create_tf_record(train_output_dir, train_dir, train_examples)
  create_tf_record(val_output_dir, valid_dir, val_examples)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
