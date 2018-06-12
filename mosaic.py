import argparse
import cv2

import os

import numpy as np
from random import randint
from params import Params

index_matrix = None


def compute_dimensions():
    # compute dimensions of generate image,
    # based of image scale and Params.number_mosaic_parts_horizontal
    [pieseH, pieseW, _] = \
        np.shape(Params.mosaic_pieces[0])
    [refH, refW, _] = np.shape(Params.refImage)

    rezW = Params.number_mosaic_parts_horizontal * pieseW
    scara = rezW / refW
    rezH = refH * scara

    Params.number_mosaic_parts_vertical = int(np.floor(rezH / pieseH))
    H = Params.number_mosaic_parts_vertical * pieseH
    W = Params.number_mosaic_parts_horizontal * pieseW
    Params.ref_image_resized = cv2.resize(Params.refImage, (W, H))


def add_mosaic_piece_krodo():
    global index_matrix
    img_mosaic = np.uint8(np.zeros(np.shape(Params.ref_image_resized)))
    index_matrix = [[-1 for x in range(Params.number_mosaic_parts_horizontal + 2)] for y in
                    range(Params.number_mosaic_parts_vertical + 2)]
    index_matrix = np.asarray(index_matrix)

    img_mosaic = np.asarray(img_mosaic)

    [N, H, W, C] = np.shape(Params.mosaic_pieces)
    [h, w, c] = np.shape(Params.ref_image_resized)

    total_pieces_no = Params.number_mosaic_parts_horizontal * Params.number_mosaic_parts_vertical
    add_pieces_no = 0

    avg_color_list = compute_average_color_list()
    for i in range(1, Params.number_mosaic_parts_vertical + 1):
        for j in range(1, Params.number_mosaic_parts_horizontal + 1):

            if Params.identical_matching_pieces == 1:
                indice = find_average_color(
                    Params.ref_image_resized[(i - 1) * H: i * H, (j - 1) * W: j * W, :]
                    , avg_color_list)
            else:
                indice = find_average_color(
                    Params.ref_image_resized[(i - 1) * H: i * H, (j - 1) * W: j * W, :]
                    , avg_color_list, i - 1, j - 1)

            img_mosaic[(i - 1) * H: i * H, (j - 1) * W: j * W, :] = \
                Params.mosaic_pieces[indice][:][:][:]

            add_pieces_no += 1

        print("Building mosaic. {}% ready ".format(int(100 * add_pieces_no / total_pieces_no)))

    return img_mosaic


def add_mosaic_piece_random_position():
    img_mosaic = np.uint8(np.zeros(np.shape(Params.ref_image_resized)))

    img_mosaic = np.asarray(img_mosaic)

    [N, H, W, C] = np.shape(Params.mosaic_pieces)
    [h, w, c] = np.shape(Params.ref_image_resized)

    add_pieces_no = 0

    avg_color_list = compute_average_color_list()

    crossings_no = Params.crossings_no

    rangeH = h - H
    rangeW = w - W

    if Params.rand_criterion == 'try_hard':

        empty_mozaic = {(row * rangeW + col): (row, col) for row in range(0, rangeH) for col in range(0, rangeW)}

        pixels_to_fill = rangeH * rangeW

        while empty_mozaic:

            # get a random block to fill from the empty blocks list
            randomIndex = randint(0, pixels_to_fill)

            if randomIndex not in empty_mozaic:
                continue

            (i, j) = empty_mozaic[randomIndex]

            index = find_average_color(Params.ref_image_resized
                                        [i: i + H, j: j + W, :], avg_color_list)

            img_mosaic[i: i + H, j: j + W, :] = \
                Params.mosaic_pieces[index][:][:][:]

            # mark the previously empty block as full
            for row in range(i, (i + H + 1)):
                for col in range(j, (j + W + 1)):
                    if randomIndex in empty_mozaic:
                        del (empty_mozaic[randomIndex])

            print("Building mosaic. {} pieces to add".format(len(empty_mozaic)))

    elif Params.rand_criterion == 'stochastic':

        total_pieces_no = Params.number_mosaic_parts_horizontal * Params.number_mosaic_parts_vertical

        total_generated_no = total_pieces_no * crossings_no

        for i in range(0, total_generated_no):
            i = randint(0, rangeH)
            j = randint(0, rangeW)

            index = find_average_color(Params.ref_image_resized
                                        [i: i + H, j: j + W, :], avg_color_list)

            img_mosaic[i: i + H, j: j + W, :] = \
                Params.mosaic_pieces[index][:][:][:]

            add_pieces_no += 1

            print("Building mosaic. {}% ready", format(int(100 * add_pieces_no / total_generated_no)))

    return img_mosaic


def find_average_color(img, avg_img_list, index_h=-1, index_w=-1):
    _index = 1
    distance = float('inf')

    average_color_img = [img[:, :, i].mean() for i in range(img.shape[-1])]

    if index_w == -1 & index_h == -1:
        id = 0
        for avg_color in avg_img_list:

            dist = np.linalg.norm(np.array(average_color_img) - np.array(avg_color))
            if dist < distance:
                _index = id
                distance = dist
            id += 1
        return _index
    else:
        id = 0
        for avg_color in avg_img_list:
            dist = np.linalg.norm(np.array(average_color_img) - np.array(avg_color))

            if dist < distance:

                if index_h == 0 and index_w == 0:
                    _index = id
                    distance = dist
                if index_h > 0 and index_w == 0 and index_matrix[index_h - 1][index_w] != id:

                    _index = id
                    distance = dist
                elif index_h == 0 and index_w > 0 \
                        and index_matrix[index_h][index_w - 1] != id:
                    _index = id
                    distance = dist
                elif index_h > 0 and index_w > 0 \
                        and index_matrix[index_h - 1][index_w] != id \
                        and index_matrix[index_h][index_w - 1] != id:

                    _index = id
                    distance = dist

            id += 1

        index_matrix[index_h][index_w] = _index

        return _index


def compute_average_color_list():
    [N, _, _, _] = np.shape(Params.mosaic_pieces)
    average_color = []

    for i in range(0, N - 1):
        image = Params.mosaic_pieces[i][:][:][:]
        avg_color = [image[:, :, i].mean() for i in range(image.shape[-1])]

        average_color.append(avg_color)

    return average_color


def load_mosaic_parts():
    print('Load mosaic pieces into memory')

    dirPath = os.path.join('dataset/cifar-images/', str(Params.category))

    files = os.listdir(dirPath)

    im_piese = cv2.imread(dirPath + '/' + files[0])

    [H, W, C] = np.shape(im_piese)
    mosaic_pieces = np.zeros([len(files), H, W, C], np.uint8)
    index = 0

    for myFile in files:
        image = cv2.imread(dirPath + '/' + myFile)
        assert np.shape(image) == (H, W, C), "img %s has shape %r" % (myFile, image.shape)

        image = np.asarray(image)

        mosaic_pieces[index][:][:][:] = image

        index += 1

    Params.mosaic_pieces = mosaic_pieces


def add_mosaic_piece():
    global index_matrix
    img_mosaic = np.uint8(np.zeros(np.shape(Params.ref_image_resized)))
    index_matrix = [[-1 for x in range(Params.number_mosaic_parts_horizontal + 2)] for y in
                    range(Params.number_mosaic_parts_vertical + 2)]
    index_matrix = np.asarray(index_matrix)

    img_mosaic = np.asarray(img_mosaic)

    [_, H, W, _] = np.shape(Params.mosaic_pieces)
    [h, w, c] = np.shape(Params.ref_image_resized)

    total_pieces_no = Params.number_mosaic_parts_horizontal * Params.number_mosaic_parts_vertical
    add_pieces_no = 0

    avgColorList = compute_average_color_list()
    for i in range(1, Params.number_mosaic_parts_vertical + 1):
        for j in range(1, Params.number_mosaic_parts_horizontal + 1):

            if Params.identical_matching_pieces == 1:
                indice = find_average_color(
                    Params.ref_image_resized[(i - 1) * H: i * H, (j - 1) * W: j * W, :]
                    , avgColorList)
            else:
                indice = find_average_color(
                    Params.ref_image_resized[(i - 1) * H: i * H, (j - 1) * W: j * W, :]
                    , avgColorList, i - 1, j - 1)

            img_mosaic[(i - 1) * H: i * H, (j - 1) * W: j * W, :] = \
                Params.mosaic_pieces[indice][:][:][:]

            add_pieces_no = add_pieces_no + 1

        print("Building mosaic. {}% ready ".format(int(100 * add_pieces_no / total_pieces_no)))

    return img_mosaic


def add_mosaic_piece_random():
    img_mosaic = np.uint8(np.zeros(np.shape(Params.ref_image_resized)))

    img_mosaic = np.asarray(img_mosaic)

    [N, H, W, C] = np.shape(Params.mosaic_pieces)
    [h, w, c] = np.shape(Params.ref_image_resized)

    added_piece_no = 0

    avg_color_list = compute_average_color_list()

    crossing_no = Params.crossings_no

    rangeH = h - H
    rangeW = w - W

    if Params.rand_criterion == 'try_hard':

        empty_mosaic = {(row * rangeW + col): (row, col) for row in range(0, rangeH) for col in range(0, rangeW)}

        pixels_to_fill = rangeH * rangeW

        while empty_mosaic:

            # get a random block to fill from the empty blocks list
            randomIndex = randint(0, pixels_to_fill)

            if randomIndex not in empty_mosaic:
                continue

            (i, j) = empty_mosaic[randomIndex]

            indice = find_average_color(Params.ref_image_resized
                                        [i: i + H, j: j + W, :], avg_color_list)

            img_mosaic[i: i + H, j: j + W, :] = \
                Params.mosaic_pieces[indice][:][:][:]

            # mark the previously empty block as full
            for row in range(i, (i + H + 1)):
                for col in range(j, (j + W + 1)):
                    if randomIndex in empty_mosaic:
                        del (empty_mosaic[randomIndex])

            print("Building mosaic. {} pieces left".format(len(empty_mosaic)))

    elif Params.rand_criterion == 'stochastic':

        total_no_pieces = Params.number_mosaic_parts_horizontal * Params.number_mosaic_parts_vertical

        total_gen_number = total_no_pieces * crossing_no

        for i in range(0, total_gen_number):
            i = randint(0, rangeH)
            j = randint(0, rangeW)

            indice = find_average_color(Params.ref_image_resized
                                        [i: i + H, j: j + W, :], avg_color_list)

            img_mosaic[i: i + H, j: j + W, :] = \
                Params.mosaic_pieces[indice][:][:][:]

            added_piece_no = added_piece_no + 1

            print("Building mosaic, {}% ready".format(int(100 * added_piece_no / total_gen_number)))

    return img_mosaic


def find_average_color(img, avg_img_list, index_h=-1, index_w=-1):
    _index = 1
    distance = float('inf')

    average_color_img = [img[:, :, i].mean() for i in range(img.shape[-1])]

    if index_w == -1 & index_h == -1:
        id = 0
        for avg_color in avg_img_list:

            dist = np.linalg.norm(np.array(average_color_img) - np.array(avg_color))
            if dist < distance:
                _index = id
                distance = dist
            id += 1
        return _index
    else:
        id = 0
        for avg_color in avg_img_list:
            dist = np.linalg.norm(np.array(average_color_img) - np.array(avg_color))

            if dist < distance:

                if index_h == 0 and index_w == 0:
                    _index = id
                    distance = dist
                if index_h > 0 and index_w == 0 and index_matrix[index_h - 1][index_w] != id:

                    _index = id
                    distance = dist
                elif index_h == 0 and index_w > 0 \
                        and index_matrix[index_h][index_w - 1] != id:
                    _index = id
                    distance = dist
                elif index_h > 0 and index_w > 0 \
                        and index_matrix[index_h - 1][index_w] != id \
                        and index_matrix[index_h][index_w - 1] != id:

                    _index = id
                    distance = dist

            id += 1

        index_matrix[index_h][index_w] = _index

        return _index


# def computeAverageColorForAList():
#     [N, _, _, _] = np.shape(Params.mosaic_pieces)
#     average_color = []
#
#     for i in range(0, N - 1):
#         image = Params.mosaic_pieces[i][:][:][:]
#         avg_color = [image[:, :, i].mean() for i in range(image.shape[-1])]
#
#         average_color.append(avg_color)
#
#     return average_color


def build_mosaic():
    load_mosaic_parts()

    compute_dimensions()

    if args.arranging_way == 'krado':
        img_mosaic = add_mosaic_piece_krodo()
    elif args.arranging_way == 'random':
        img_mosaic = add_mosaic_piece_random_position()

    return img_mosaic


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to build a mosaic from pictures')
    parser.add_argument('--ref_image_path', '-ref', type=str)
    parser.add_argument('--save_path', type=str, default='mozaic.jpg')
    parser.add_argument('--number_parts_horizontal', type=int, default=50)
    parser.add_argument('--arranging_way', type=str, default='krado', choices=['krado', 'random'])
    parser.add_argument('--random_criterion', type=str, default='stochastic', choices=['stochastic', 'try_hard'])
    parser.add_argument('--crossings_no', type=int, default=2)
    parser.add_argument('--identical_matching_pieces', type=int, default=10)
    parser.add_argument('--category', type=int, default=1, choices=[x + 1 for x in range(10)])

    args = parser.parse_args()
    Params.update_params(args)

    Params.refImage = cv2.imread(Params.ref_image_path, cv2.IMREAD_COLOR)
    img_mozaic = build_mosaic()

    print('Saving image to ' + args.save_path)
    cv2.imwrite(args.save_path, img_mozaic)
