import PIL
import numpy as np
import os
import argparse
import shutil
import sys
import cv2
from os import listdir
from os.path import join
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument('--root', type=str, default='./road/raw_data',
                    help='Path to the directory containing the road data.')

parser.add_argument('--target_dir', type=str, default='./road/data',
                    help='Path to the directory containing the road data.')


def to_binary(img):
    """ RGB to binary (0, 255) """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return im_bw


def shuffle_in_pair(images, masks):
    """ Random shuffle """
    new_images = np.copy(images)
    new_masks = np.copy(masks)
    permutation = np.random.permutation(len(new_images))
    for old_index, new_index in enumerate(permutation):
        new_images[new_index] = images[old_index]
        new_masks[new_index] = masks[old_index]
    return new_images, new_masks


def match(images, masks):
    images.sort()
    masks.sort()
    img_index = 0
    mask_index = 0
    new_images, new_masks = [], []
    while img_index < len(images) and mask_index < len(masks):
        x = int(images[img_index].split('_')[0])
        y = int(masks[mask_index].split('_')[0])
        if x == y:
            new_images.append(images[img_index])
            new_masks.append(masks[mask_index])
            img_index += 1
            mask_index += 1
        elif x < y:
            img_index += 1
        else:
            mask_index += 1

    return new_images, new_masks


def separate(target_dir, percentage, save_to_folder=True):
    """ Create train.txt and val.txt """
    images = [f for f in listdir(target_dir) if f.endswith('.jpg')]
    masks = [f for f in listdir(target_dir) if f.endswith('.png')]
    images, masks = match(images, masks)
    # shuffle before separated
    images, masks = shuffle_in_pair(images, masks)

    X_train = images[0: int(len(images) * percentage)]
    Y_train = masks[0: int(len(masks) * percentage)]
    X_valid = images[int(len(images) * percentage):]
    Y_valid = masks[int(len(masks) * percentage):]

    trainfile = open(join(target_dir, "train.txt"), "w")
    for i, l in zip(X_train, Y_train):
        trainfile.write(i + " " + l + "\n")

    testfile = open(join(target_dir, "valid.txt"), "w")
    for i, l in zip(X_valid, Y_valid):
        testfile.write(i + " " + l + "\n")

    # save train and valid into folders
    if save_to_folder:
        train_dir = join(target_dir, "train")
        valid_dir = join(target_dir, "valid")
        os.mkdir(train_dir)
        os.mkdir(valid_dir)

        for x in X_train:
            if x.endswith('.jpg'):
                shutil.move(join(target_dir, x), join(train_dir, x))
        for y in Y_train:
            if y.endswith('.png'):
                shutil.move(join(target_dir, y), join(train_dir, y))
        for x in X_valid:
            if x.endswith('.jpg'):
                shutil.move(join(target_dir, x), join(valid_dir, x))
        for y in Y_valid:
            if y.endswith('.png'):
                shutil.move(join(target_dir, y), join(valid_dir, y))

    trainfile.close()
    testfile.close()


def main(unused_argv):
    tf.logging.info("Reading from road train_val dataset...")

    img_dir = join(FLAGS.root, 'train_val')
    target_dir = FLAGS.target_dir
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    files = [f for f in listdir(img_dir) if f.endswith(('.png', '.jpg'))]

    for file in files:
        img = cv2.imread(join(img_dir, file))
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        cv2.imwrite(join(target_dir, file), img)
        if file.endswith('png'):
            img = to_binary(img)
            img = img // 255
            cv2.imwrite(join(target_dir, file), img)

    separate(target_dir, 0.9)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
