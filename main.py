import pprint

import numpy
from PIL import Image
import numpy as np
from scipy.fft import dct, idct

C = np.array([
    [0.354, 0.354, 0.354, 0.354, 0.354, 0.354, 0.354, 0.354],
    [0.490, 0.416, 0.278, 0.098, -0.098, -0.278, -0.416, -0.490],
    [0.462, 0.191, -0.191, -0.462, -0.462, -0.191, 0.191, 0.462],
    [0.416, -0.098, -0.490, -0.278, 0.278, 0.490, 0.098, -0.416],
    [0.354, -0.354, -0.354, 0.354, 0.354, -0.354, -0.354, 0.354],
    [0.278, -0.490, 0.098, 0.416, -0.416, -0.098, 0.490, -0.378],
    [0.191, -0.462, 0.462, -0.191, -0.191, 0.462, -0.462, 0.191],
    [0.098, -0.278, 0.416, -0.490, 0.490, -0.416, 0.278, -0.098]
])


def diagonal_traversal(matrix):
    if not matrix or not matrix[0]:
        return []

    rows, cols = len(matrix), len(matrix[0])
    result = []
    intermediate = []

    for d in range(rows + cols - 1):
        intermediate.clear()

        r = 0 if d < cols else d - cols + 1
        c = d if d < cols else cols - 1

        while r < rows and c > -1:
            intermediate.append(matrix[r][c])
            r += 1
            c -= 1

        if d % 2 == 0:
            result.extend(intermediate[::-1])
        else:
            result.extend(intermediate)
    return result


def decrease_pixels(matrix: numpy.array) -> numpy.array:
    for line in range(len(matrix)):
        for row in range(len(matrix[line])):
            matrix[line][row] -= 128
    return matrix


def arnold_transformation(matrix) -> numpy.array:
    n = len(matrix)
    new_matrix = matrix.copy()
    for i in range(n):
        for j in range(n):
            new_cords = np.matmul(np.array([[1, 1], [1, 2]]), np.array([[i], [j]]))
            i_new, j_new = new_cords[0][0] % n, new_cords[1][0] % n
            new_matrix[i_new][j_new] = matrix[i][j]
    return new_matrix


def arnold_regeneration(matrix: numpy.array) -> numpy.array:
    n, _ = matrix.shape
    old_matrix = matrix.copy()
    for i in range(n):
        for j in range(n):
            old_cords = numpy.matmul(np.array([[2, -1], [-1, 1]]), np.array([[i], [j]]))
            i_old, j_old = old_cords[0][0] % n, old_cords[1][0] % n
            old_matrix[i_old][j_old] = matrix[i][j]
    return old_matrix


def get_bin_matrix_of_wm():
    wm_image = Image.open("watermark.png")
    wm_matrix = np.array(wm_image)
    wm_bin_matrix = [[None for _ in range(wm_matrix.shape[1])] for _ in range(wm_matrix.shape[0])]
    for i in range(len(wm_matrix)):
        for j in range(len(wm_matrix[i])):
            wm_bin_matrix[i][j] = 1 if wm_matrix[i][j][0] > 127 else 0
    return wm_bin_matrix


# def split_8(matrix):
#     n = len(matrix)
#     arr = []
#     for block_row in range(n // 8):
#         for block_col in range(n // 8):
#             arr.append([[matrix[i][j] for j in range(block_col * (n // 8), (block_col + 1) * (n // 8))]
#                         for i in range(block_row * (n // 8), (block_row + 1) * (n // 8))])
#     return arr


def split_into_blocks_by_8(matrix):
    blocks = []
    for i in range(0, len(matrix), 8):
        for j in range(0, len(matrix[i]), 8):
            block = [row[j:j + 8] for row in matrix[i:i + 8]]
            blocks.append(block)
    return blocks


# def merge_8(blocks):
#     result = [[0 for _ in range(len(blocks[0]) * 8)] for _ in range(len(blocks[0]) * 8)]
#     for ind in range(len(blocks)):
#         block_row = ind // 8
#         block_col = ind % 8
#         block = blocks[ind]
#         for i in range(block_row * 8, (block_row + 1) * 8):
#             for j in range(block_col * 8, (block_col + 1) * 8):
#                 result[i][j] = block[i - block_row * 8][j - block_col * 8]
#     return result


def merge_blocks_by_8(blocks):
    block_count = len(blocks)
    block_size = len(blocks[0])
    dim = int((block_count * block_size) ** 0.5)

    matrix = [[0] * dim for _ in range(dim)]

    for b in range(block_count):
        for i in range(block_size):
            for j in range(block_size):
                row = b // (dim // block_size) * block_size + i
                col = b % (dim // block_size) * block_size + j
                matrix[row][col] = blocks[b][i][j]
    return matrix


def modification(block):
    z = 2
    snaked = diagonal_traversal(block)
    dc = snaked[0]
    med = sorted(snaked[1:10])[4]
    return abs(z * ((dc - med) / dc)) if 1 <= abs(dc) <= 1000 else abs(z * med)


def insert(matrix: numpy.array) -> numpy.array:
    matrix = decrease_pixels(matrix)
    blocks = split_into_blocks_by_8(matrix)
    wm_bits = arnold_transformation(get_bin_matrix_of_wm())
    # print(blocks[0])
    # print(dct(blocks[0]))
    # print(blocks[0])
    for i in range(len(blocks)):
        after_dct = dct(blocks[i])
        M = modification(after_dct)


# m = get_bin_matrix_of_wm()
# wm_img = Image.open("watermark.png")
# m_new = np.array(wm_img)
# for i in range(len(m)):
#     for j in range(len(m[i])):
#         m[i][j] = [255, 255, 255, 255] if m[i][j] else [0, 0, 0, 255]
# Image.fromarray(np.array(m).astype("uint8")).save("watermark_bit.png")


# uint8


if __name__ == '__main__':
    img = Image.open("JasehOnfroyChild.png")
    img_matrix = np.array(img)
    insert(img_matrix)
