import argparse

import numpy as np
import cv2


def compute_energy(image):
    ddept = cv2.CV_64F

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = np.array(gray_img)

    sobelx = cv2.Sobel(gray_img, ddept, 1, 0)
    sobely = cv2.Sobel(gray_img, ddept, 0, 1)

    dxabs = cv2.convertScaleAbs(sobelx)
    dyabs = cv2.convertScaleAbs(sobely)

    energy_matrix = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)

    return energy_matrix


def select_vertical_path(energy_matrix):
    d = np.int32(np.zeros([np.shape(energy_matrix)[0], 2]))

    cost_matrix = np.empty_like(energy_matrix)

    cost_matrix = cost_matrix.astype(np.int32, copy=False)

    cost_matrix[0, :] = energy_matrix[0, :]

    for i in range(1, np.shape(energy_matrix)[0]):
        for j in range(0, np.shape(energy_matrix)[1]):

            if j == 0:
                cost_matrix[i][j] = energy_matrix[i, j] + min([cost_matrix[i - 1][j], cost_matrix[i - 1][j + 1]])
            elif j == np.shape(energy_matrix)[1] - 1:
                cost_matrix[i][j] = energy_matrix[i, j] + min([cost_matrix[i - 1][j - 1], cost_matrix[i - 1][j]])
            else:
                cost_matrix[i][j] = energy_matrix[i, j] + min(
                    [cost_matrix[i - 1][j - 1],
                     cost_matrix[i - 1][j],
                     cost_matrix[i - 1][j + 1]])

    line = np.shape(energy_matrix)[0] - 1
    column = 0
    min_magnitude = cost_matrix[line][0]

    for index in range(0, np.shape(energy_matrix)[1]):
        if cost_matrix[line][index] < min_magnitude:
            min_magnitude = cost_matrix[line][index]
            column = index
    d[line, :] = [line, column]

    for i in range(np.shape(energy_matrix)[0] - 2, -1, -1):

        line = i

        if d[i + 1, 1] == 0:
            if cost_matrix[i, 0] <= cost_matrix[i, 1]:
                optiune = 0
            else:
                optiune = 1
        elif d[i + 1, 1] == np.shape(energy_matrix)[1] - 1:

            if cost_matrix[i, d[i + 1, 1]] <= cost_matrix[i, d[i + 1, 1] - 1]:
                optiune = 0
            else:
                optiune = -1
        else:
            list = [cost_matrix[i, d[i + 1, 1] - 1],
                    cost_matrix[i, d[i + 1, 1]],
                    cost_matrix[i,
                                d[i + 1, 1] + 1]]

            if cost_matrix[i, d[i + 1, 1]] == min(list):
                optiune = 0
            else:
                indexMin = list.index(min(list))
                optiune = indexMin - 1

        column = d[i + 1, 1] + optiune

        d[i, :] = [line, column]

    return d


def select_horozontal_path(energy_matrix):
    d = np.int32(np.zeros([np.shape(energy_matrix)[1], 2]))

    cost_matrix = np.empty_like(energy_matrix)
    cost_matrix = cost_matrix.astype(np.int32, copy=False)
    cost_matrix[:, 0] = energy_matrix[:, 0]

    for j in range(1, np.shape(energy_matrix)[1]):
        for i in range(0, np.shape(energy_matrix)[0]):
            if i == 0:
                cost_matrix[i][j] = energy_matrix[i, j] + min([cost_matrix[i][j - 1], cost_matrix[i + 1][j - 1]])
            elif i == np.shape(energy_matrix)[0] - 1:
                cost_matrix[i][j] = energy_matrix[i, j] + min([cost_matrix[i][j - 1], cost_matrix[i - 1][j - 1]])
            else:
                cost_matrix[i][j] = energy_matrix[i, j] + min(
                    [cost_matrix[i - 1][j - 1],
                     cost_matrix[i][j - 1],
                     cost_matrix[i + 1][j - 1]])

    column = np.shape(energy_matrix)[1] - 1
    line = 0
    min_magnitude = cost_matrix[line][column]

    for index in range(0, np.shape(energy_matrix)[0]):
        if cost_matrix[index][column] < min_magnitude:
            min_magnitude = cost_matrix[index][column]
            line = index
    d[line, :] = [line, column]

    for i in range(np.shape(energy_matrix)[1] - 2, 0, -1):

        column = i

        if d[i + 1, 0] == 0:
            if cost_matrix[0, i] <= cost_matrix[1, i]:
                option = 0
            else:
                option = 1
        elif d[i + 1, 0] == np.shape(energy_matrix)[0] - 1:

            if cost_matrix[d[i + 1, 0], i] <= cost_matrix[d[i + 1, 0] - 1, i]:
                option = 0
            else:
                option = -1
        else:
            list = [cost_matrix[d[i + 1, 0] - 1, i],
                    cost_matrix[d[i + 1, 0], i],
                    cost_matrix[d[i + 1, 0] + 1, i]]
            indexMin = list.index(min(list))
            option = indexMin - 1

        line = d[i + 1, 0] + option

        d[i, :] = [line, column]

    return d


def insert_horizontal_path(image, path):
    img1 = np.uint8(np.zeros((np.shape(image)[0] + 1, np.shape(image)[1], np.shape(image)[2])))

    for i in range(0, np.shape(img1)[1]):
        line = path[i, 0]

        img1[0:line, i, :] = image[0:line, i, :]

        if line == 0:
            left_path = image[line, i, :]
            right_path = np.round((image[line, i, :].astype(int) + image[line + 1, i, :].astype(int)) / 2)
        elif line == np.shape(image)[0] - 1:
            left_path = np.round((image[line - 1, i, :].astype(int) + image[line, i, :].astype(int)) / 2)
            right_path = image[line, i, :]
        else:
            left_path = np.round((image[line - 1, i, :].astype(int) + image[line, i, :].astype(int)) / 2)
            right_path = np.round((image[line, i, :].astype(int) + image[line + 1, i, :].astype(int)) / 2)
        img1[line, i, :] = left_path
        img1[line + 1, i, :] = right_path

        img1[line + 2:, i, :] = image[line + 1:, i, :]
    return img1


def insert_vertical_path(image, path):
    image = np.uint8(np.zeros((np.shape(image)[0], np.shape(image)[1] + 1, np.shape(image)[2])))

    for i in range(0, np.shape(image)[0]):
        column = path[i, 1]

        image[i, 0:column, :] = image[i, 0:column, :]

        if column == 0:
            left_path = image[i, column, :]
            right_path = np.round((image[i, column, :].astype(int) + image[i, column + 1, :].astype(int)) / 2)
        elif column == np.shape(image)[1] - 1:
            left_path = np.round((image[i, column - 1, :].astype(int) + image[i, column, :].astype(int)) / 2)
            right_path = image[i, column, :]
        else:
            left_path = np.round((image[i, column - 1, :].astype(int) + image[i, column, :].astype(int)) / 2)
            right_path = np.round((image[i, column, :].astype(int) + image[i, column + 1, :].astype(int)) / 2)

        image[i, column, :] = left_path
        image[i, column + 1, :] = right_path

        image[i, column + 2:, :] = image[i, column + 1:, :]

    return image


def cut_vertical_path(image, path):
    resized_image = np.uint8(np.zeros((np.shape(image)[0], np.shape(image)[1] - 1, np.shape(image)[2])))

    for i in range(0, np.shape(resized_image)[0]):
        column = path[i][1]

        # TODO : fix coloana = 0
        if column != 0:
            resized_image[i, 0:column, :] = image[i, 0:column, :]
            resized_image[i, column:, :] = image[i, column + 1:, :]
        elif column == 0:
            resized_image[i, :, :] = image[i, 1:, :]

    return resized_image


def cut_horizontal_path(image, path):
    img1 = np.uint8(np.zeros((np.shape(image)[0] - 1, np.shape(image)[1], np.shape(image)[2])))

    for i in range(0, np.shape(img1)[1]):
        line = path[i][0]

        # TODO : fix linia = 0
        if line != 0:
            img1[0:line, i, :] = image[0:line, i, :]
            img1[line:, i, :] = image[line + 1:, i, :]
        else:
            img1[line:, i, :] = image[1:, i, :]

    return img1


def enlarge_height(image, pixel_height_no):
    _, paths = shrink_height(image, pixel_height_no)
    img1 = image
    v = np.zeros([pixel_height_no])

    x = np.zeros([pixel_height_no])

    for i in range(0, pixel_height_no):
        path = paths[i]
        v[i] = \
            path[0, 0]

    x[0] = 0

    for i in range(1, pixel_height_no):
        count = 0
        for j in range(0, i):
            if v[i] > v[j]:
                count += 1
        x[i] = count

    for i in range(0, pixel_height_no):
        print('Adding seam. Seam to add : {}'.format(pixel_height_no - i))

        path = paths[i]
        path[:, 0] = path[:, 0] + 2 * x[i]

        img1 = insert_horizontal_path(img1, path)

    return img1


def enlarge_width(image, width_pixel_no):
    _, paths = shrink_width(image, width_pixel_no)

    img1 = image

    v = np.zeros([width_pixel_no])

    x = np.zeros([width_pixel_no])

    for i in range(0, width_pixel_no):
        path = paths[i]
        v[i] = \
            path[0, 1]

    x[0] = 0
    for i in range(1, width_pixel_no):
        count = 0
        for j in range(0, i):
            if v[i] > v[j]:
                count += 1
        x[i] = count

    for i in range(0, width_pixel_no):
        print('Adding seam. Seam to add : {}'.format(width_pixel_no - i))

        path = paths[i]
        path[:, 1] = path[:, 1] + 2 * x[i]

        img1 = insert_vertical_path(img1, path)

    return img1


def shrink_width(image, width_pixel_no):
    paths = []
    for i in range(0, width_pixel_no):
        print('Paths left : {}'.format(width_pixel_no - i))

        energy_matrix = compute_energy(image)

        paths = select_vertical_path(energy_matrix)
        paths.append(paths)

        image = cut_vertical_path(image, paths)

    return image, paths


def shrink_height(image, height_pixel_no):
    paths = []
    for i in range(0, height_pixel_no):
        print('Paths left : {}'.format(height_pixel_no - i))

        E = compute_energy(image)

        path = select_horozontal_path(E)
        paths.append(path)

        image = cut_horizontal_path(image, path)

    return image, paths


def remove_object(image):
    showCrosshair = False
    fromCenter = False
    r = cv2.selectROI("Image", image, fromCenter, showCrosshair)

    if r[2] < r[3]:

        paths_no = r[2]
        while paths_no > 0:
            print("Removing seams. Paths left {}".format(paths_no))

            energy_matrix = compute_energy(image)

            energy_matrix = energy_matrix.astype(np.float32)
            energy_matrix[r[1]:r[1] + r[3], r[0]:r[0] + paths_no] = -9999

            paths_no -= 1

            path = select_vertical_path(energy_matrix)

            image = cut_vertical_path(image, path)
    else:

        paths_no = r[3]
        while paths_no > 0:
            print("Removing seams. Paths left {}".format(paths_no))
            energy_matrix = compute_energy(image)
            energy_matrix = energy_matrix.astype(np.float32)
            energy_matrix[r[1]:r[1] + paths_no, r[0]:r[0] + r[2]] = -9999

            # print E[r[1]:r[1] + r[3], r[0]:r[0] + x]

            paths_no -= 1

            path = select_horozontal_path(energy_matrix)

            image = cut_horizontal_path(image, path)

    return image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to seam carve a photo')
    parser.add_argument('--ref_image_path', '-ref', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='seam.jpg')
    parser.add_argument('--resize_option', '-opt', type=str, default='remove_object',
                        choices=['shrink_width', 'shrink_height', 'enlarge_width', 'enlarge_height', 'remove_object'])
    parser.add_argument('--width_pixel_no', type=int, default=100)
    parser.add_argument('--height_pixel_no', type=int, default=50)

    args = parser.parse_args()

    img = cv2.imread(args.ref_image_path)
    img = np.uint8(img)

    if args.resize_option == 'shrink_width':
        resize_img, _ = shrink_width(img, args.width_pixel_no)

    elif args.resize_option == 'shrink_height':
        resize_img, _ = shrink_height(img, args.height_pixel_no)

    elif args.resize_option == 'enlarge_width':
        resize_img = enlarge_width(img, args.width_pixel_no)

    elif args.resize_option == 'enlarge_height':
        resize_img = enlarge_height(img, args.height_pixel_no)

    elif args.resize_option == 'remove_object':
        resize_img = remove_object(img)
    else:
        raise ValueError('Resize option not known')

    print('Image saved to {}'.format(args.save_path))
    cv2.imwrite(args.save_path, resize_img)
