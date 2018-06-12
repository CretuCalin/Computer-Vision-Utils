import mxnet as mx
import numpy as np
import _pickle as cPickle
import cv2
import os


def extract_images_and_labels(path, file):
    f = open(path + file, 'rb')
    dict = cPickle.load(f, encoding='latin1')
    images = dict['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = dict['labels']
    imagearray = mx.nd.array(images)
    labelarray = mx.nd.array(labels)
    return imagearray, labelarray


def extract_categories(path, file):
    f = open(path + file, 'rb')
    dict = cPickle.load(f)
    return dict['label_names']


def save_cifar_image(array, path, file):
    # array is 3x32x32. cv2 needs 32x32x3
    array = array.asnumpy().transpose(1, 2, 0)
    # array is RGB. cv2 needs BGR
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    # save to PNG file
    return cv2.imwrite(path + file + ".png", array)


if __name__ == '__main__':

    imgarray, lblarray = extract_images_and_labels("dataset/cifar-10-batches-py/", "data_batch_1")

    categories = extract_categories("dataset/cifar-10-batches-py/", "batches.meta")
    print("Categories : ", categories)
    categories_labels = []

    if not os.path.exists('dataset/cifar-images'):
        os.makedirs('dataset/cifar-images')

    for i in range(len(categories)):
        if not os.path.exists(os.path.join('dataset/cifar-images', str(i + 1))):
            os.makedirs(os.path.join('dataset/cifar-images', str(i + 1)))

    for i in range(0, imgarray.shape[0]):
        category = lblarray[i].asnumpy()
        category = int(category[0])
        categories_labels.append(categories[category])
        save_cifar_image(imgarray[i], "dataset/cifar-images/" + str(category + 1) + "/", "image" + str(i))

    print('Images extracted')
