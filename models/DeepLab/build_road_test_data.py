import math
import os.path
import sys
import build_data
import tensorflow as tf

from os import listdir
from os.path import join


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './road/raw_data/test', 'Folder containing images.')

tf.app.flags.DEFINE_string('output_dir',  './road/tfrecord', 'Path to tfrecord.')

_NUM_SHARDS = 1


def _convert_dataset():
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  images = [f for f in listdir(FLAGS.data_dir) if f.endswith('.jpg')]
  num_images = len(images)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(FLAGS.output_dir, 'test-%05d-of-%05d.tfrecord' % (shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(images), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(FLAGS.data_dir, images[i])
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = label_reader.read_image_dims(image_data)

        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(image_data, images[i], height, width)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def main(unused_argv):
    _convert_dataset()


if __name__ == '__main__':
  tf.app.run()
