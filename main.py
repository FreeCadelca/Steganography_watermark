import pprint
import sys

import numpy
from PIL import Image
import numpy as np
from scipy.fft import dct, idct
import math

count_of_small_m = 0


def my_dct_by_blocks(matrix):
    def Cf(var):
        return 1 / (2 ** 0.5) if var == 0 else 1

    dcp_matrix = [[[0, 0, 0] for j in range(len(matrix[i]))] for i in range(len(matrix))]

    for block_start_i in range(0, len(matrix), 8):
        for block_start_j in range(0, len(matrix[0]) - 8, 8):
            for i_in_block in range(8):
                for j_in_block in range(8):
                    for channel in range(3):
                        c = Cf(i_in_block) * Cf(j_in_block) / 4
                        for x in range(8):
                            for y in range(8):
                                dcp_matrix[block_start_i + i_in_block][block_start_j + j_in_block][channel] += (
                                        c * math.cos((2 * x + 1) * i_in_block * math.pi / 16) *
                                        math.cos((2 * y + 1) * j_in_block * math.pi / 16) *
                                        matrix[block_start_i + x][block_start_j + y][channel])
    return np.array(dcp_matrix)


def my_idct_by_blocks(matrix):
    def Cf(var):
        return 1 / (2 ** 0.5) if var == 0 else 1

    dcp_array = matrix.copy()

    # new_matrix = [[[0, 0, 0] for j in range(len(matrix[i]))] for i in range(len(matrix))]
    # for x in range(8):
    #     for y in range(8):
    #         for channel in range(3):
    #             for u in range(8):
    #                 for v in range(8):
    #                     new_matrix[x][y][channel] += 1 / 4 * Cf(u) * Cf(v) * dcp_array[u][v][channel] * math.cos(
    #                         (2 * x + 1) * u * math.pi / 16) * math.cos((2 * y + 1) * v * math.pi / 16)

    new_matrix = [[[0, 0, 0] for _ in range(len(matrix[i]))] for i in range(len(matrix))]
    for block_start_i in range(0, len(dcp_array), 8):
        for block_start_j in range(0, len(dcp_array[0]) - 8, 8):
            for i_in_block in range(8):
                for j_in_block in range(8):
                    for channel in range(3):
                        for u in range(8):
                            for v in range(8):
                                new_matrix[block_start_i + i_in_block][block_start_j + j_in_block] += (
                                        1 / 4 * Cf(u) * Cf(v) *
                                        dcp_array[block_start_i + u][block_start_j + v] *
                                        math.cos((2 * i_in_block + 1) * u * math.pi / 16) *
                                        math.cos((2 * j_in_block + 1) * v * math.pi / 16))
    return np.array(new_matrix)


def my_dct_full_matrix(matrix):
    new_matrix = [[[0, 0, 0] for _ in range(len(matrix[i]))] for i in range(len(matrix))]
    for block_start_i in range(0, len(matrix), 8):
        for block_start_j in range(0, len(matrix[0]), 8):
            for channel in range(3):
                block = [[0] * 8 for _ in range(8)]
                for i in range(8):
                    for j in range(8):
                        block[i][j] = matrix[block_start_i + i][block_start_j + j][channel]
                block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                for i in range(8):
                    for j in range(8):
                        new_matrix[block_start_i + i][block_start_j + j][channel] = block[i][j]
    return new_matrix


def my_idct_full_matrix(matrix):
    new_matrix = [[[0, 0, 0] for _ in range(len(matrix[i]))] for i in range(len(matrix))]
    for block_start_i in range(0, len(matrix), 8):
        for block_start_j in range(0, len(matrix[0]), 8):
            for channel in range(3):
                block = [[0] * 8 for _ in range(8)]
                for i in range(8):
                    for j in range(8):
                        block[i][j] = matrix[block_start_i + i][block_start_j + j][channel]
                block = idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                for i in range(8):
                    for j in range(8):
                        new_matrix[block_start_i + i][block_start_j + j][channel] = block[i][j]
    return new_matrix


def diagonal_traversal(matrix):
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
    m_1 = matrix.copy()

    for line in range(len(m_1)):
        for row in range(len(m_1[line])):
            for channel in range(len(m_1[line][row])):
                m_1[line][row][channel] -= 128
    return m_1


def increase_pixels(matrix: numpy.array) -> numpy.array:
    m_1 = matrix.copy()

    for line in range(len(m_1)):
        for row in range(len(m_1[line])):
            for channel in range(len(m_1[line][row])):
                m_1[line][row][channel] += 128
    return m_1


def arnold_transformation(matrix) -> numpy.array:
    n = len(matrix)
    new_matrix = matrix.copy()
    for i in range(n):
        for j in range(n):
            new_cords = np.matmul(np.array([[1, 1], [1, 2]]), np.array([[i], [j]]))
            i_new, j_new = new_cords[0][0] % n, new_cords[1][0] % n
            new_matrix[i_new][j_new] = matrix[i][j]
    return new_matrix


def arnold_regeneration(matrix):
    n = len(matrix)
    old_matrix = matrix.copy()
    for i in range(n):
        for j in range(n):
            old_cords = numpy.matmul(np.array([[2, -1], [-1, 1]]), np.array([[i], [j]]))
            i_old, j_old = old_cords[0][0] % n, old_cords[1][0] % n
            old_matrix[i_old][j_old] = matrix[i][j]
    return old_matrix


def get_bin_str_of_wm():
    wm_image = Image.open("watermark.png")
    wm_matrix = np.array(wm_image)
    # wm_bin_matrix = [[None for _ in range(wm_matrix.shape[1])] for _ in range(wm_matrix.shape[0])]
    # for i in range(len(wm_matrix)):
    #     for j in range(len(wm_matrix[i])):
    #         wm_bin_matrix[i][j] = 1 if wm_matrix[i][j][0] > 127 else 0

    wm_arr = []
    for i in range(len(wm_matrix)):
        for j in range(len(wm_matrix[i])):
            wm_arr.append('1' if wm_matrix[i][j][0] > 127 else '0')
    return ''.join(wm_arr)


def split_into_blocks_by_8(matrix):
    blocks = []
    for i in range(0, len(matrix), 8):
        for j in range(0, len(matrix[i]), 8):
            block = [row[j:j + 8] for row in matrix[i:i + 8]]
            blocks.append(block)
    return blocks


def merge_blocks_by_8(blocks):
    size = int(len(blocks) ** 0.5) * 8
    matrix = [[0] * size for _ in range(size)]

    for b in range(len(blocks)):
        i_offset, j_offset = (b // (size // 8)) * 8, (b % (size // 8)) * 8
        for i in range(8):
            for j in range(8):
                matrix[i_offset + i][j_offset + j] = blocks[b][i][j]
    return matrix


def modification(block):
    z = 2
    snaked = [i[1] for i in diagonal_traversal(block)]
    dc = snaked[0]
    med = sorted(snaked[1:10])[4]
    ans = abs(z * ((dc - med) / dc)) if 1 <= abs(dc) <= 1000 else abs(z * med)
    if ans < 0.0000001:
        print("ans is too small!: ", z, dc, med)
        exponent = math.ceil(math.log10(0.00001 / ans))
        ans *= 10 ** exponent
    return ans


def new_modification(matrix, left_index_of_block, top_index_of_block, channel=1):
    i = left_index_of_block
    j = top_index_of_block
    z = 2
    dc = matrix[i][j][channel]
    ac9 = [matrix[i][j + 3][channel], matrix[i][j + 1][channel], matrix[i][j + 2][channel],
           matrix[i + 1][j][channel], matrix[i + 1][j + 1][channel], matrix[i + 1][j + 2][channel],
           matrix[i + 2][j][channel], matrix[i + 2][j + 1][channel], matrix[i + 3][j][channel]]
    med = sorted(ac9)[4]
    M = z * abs(med) if abs(dc) > 1000 or abs(dc) < 1 else z * abs(
        (matrix[i][j][channel] - med) / matrix[i][j][channel])
    # if M < 0.0000001:
    #     M *= 10 ** math.ceil(math.log10(0.00001 / M))
    if M < 0.0000001:
        M = 0.0001
        global count_of_small_m
        count_of_small_m += 1
    return M


def save_block(block):
    Image.fromarray(np.array(block).astype("uint8")).save("block.png")


def insert(matrix: numpy.array) -> numpy.array:
    T = 140
    K = 2
    matrix = decrease_pixels(matrix)
    wm_bits = get_bin_str_of_wm()
    index_of_px_in_wm = 0
    # blocks = split_into_blocks_by_8(matrix)
    # print(blocks[0])
    # print(dct(blocks[0]))
    # print(blocks[0])

    print("started dcp")
    matrix_after_dct = my_dct_full_matrix(matrix)
    print("dcp finished, started embedding")

    # for i in range(len(blocks)):
    #     if i % 1000 == 0:
    #         print(i)
    #     blocks[i] = my_dct(blocks[i])

    # debug
    # save_block(increase_pixels(blocks[15]))

    # for line in blocks[0]:
    #     for px in line:
    #         print(px[1], end='\t')
    #     print()
    n = len(matrix_after_dct[0])
    for block_start_i in range(0, len(matrix_after_dct), 8):
        for block_start_j in range(0, len(matrix_after_dct[block_start_i]), 8):
            if block_start_j == len(matrix_after_dct[block_start_i]) - 8:
                index_of_px_in_wm += 1
                continue
            M = new_modification(matrix_after_dct, block_start_i, block_start_j)
            delta = (matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -
                     matrix_after_dct[block_start_i + 4][(block_start_j + 4 + 8) % n][1])
            if wm_bits[index_of_px_in_wm] == 1:
                if delta > T - K:
                    while delta > T - K:
                        matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -= M
                        delta = (matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -
                                 matrix_after_dct[block_start_i + 4][(block_start_j + 4 + 8) % n][1])
                elif K > delta > -1 * T / 2:
                    while delta < K:
                        matrix_after_dct[block_start_i + 4][block_start_j + 4][1] += M
                        delta = (matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -
                                 matrix_after_dct[block_start_i + 4][(block_start_j + 4 + 8) % n][1])
                elif delta < -1 * T / 2:
                    while delta > 0 - T - K:
                        matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -= M
                        delta = (matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -
                                 matrix_after_dct[block_start_i + 4][(block_start_j + 4 + 8) % n][1])
            if wm_bits[index_of_px_in_wm] == 0:
                if delta > T / 2:
                    while delta <= T + K:
                        matrix_after_dct[block_start_i + 4][block_start_j + 4][1] += M
                        delta = (matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -
                                 matrix_after_dct[block_start_i + 4][(block_start_j + 4 + 8) % n][1])
                elif -1 * K < delta < T / 2:
                    while delta >= -1 * K:
                        matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -= M
                        delta = (matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -
                                 matrix_after_dct[block_start_i + 4][(block_start_j + 4 + 8) % n][1])
                elif delta < K - T:
                    while delta <= K - T:
                        matrix_after_dct[block_start_i + 4][block_start_j + 4][1] += M
                        delta = (matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -
                                 matrix_after_dct[block_start_i + 4][(block_start_j + 4 + 8) % n][1])
            index_of_px_in_wm += 1
    print("embedding finished")

    new_matrix = increase_pixels(my_idct_full_matrix(matrix_after_dct))

    # new_matrix = increase_pixels(merge_blocks_by_8(blocks))
    return new_matrix


def extract(matrix: numpy.array) -> numpy.array:
    T = 80
    K = 12
    matrix = decrease_pixels(matrix)
    blocks = split_into_blocks_by_8(matrix)
    wm_bits = []

    for i in range(len(blocks)):
        blocks[i] = dct(blocks[i])
    for i in range(len(blocks)):
        # M = modification(blocks[i])
        delta = blocks[i][4][4][1] - blocks[(i + 1) % len(blocks)][4][4][1]  # green channel
        wm_bits.append(1 if delta < -T or (0 < delta < T) else 0)
    wm_matrix = [[0] * 64 for _ in range(64)]
    for i in range(len(wm_bits)):
        wm_matrix[i // 64][i % 64] = wm_bits[i]
    # wm_matrix = arnold_regeneration(wm_matrix)
    wm_matrix_img = [[0] * 64 for _ in range(64)]
    for i in range(len(wm_matrix)):
        for j in range(len(wm_matrix[i])):
            if wm_matrix[i][j] == 1:
                wm_matrix_img[i][j] = [255, 255, 255, 255]
            else:
                wm_matrix_img[i][j] = [0, 0, 0, 255]
    out = np.array(wm_matrix_img).astype("uint8")
    return out


def extract_new(matrix):
    T = 80
    K = 12
    matrix = decrease_pixels(matrix)
    matrix_after_dct = my_dct_full_matrix(matrix)

    wm_bits = []
    n = len(matrix_after_dct[0])
    for block_start_i in range(0, len(matrix_after_dct), 8):
        for block_start_j in range(0, len(matrix_after_dct[block_start_i]), 8):
            delta = (matrix_after_dct[block_start_i + 4][block_start_j + 4][1] -
                     matrix_after_dct[block_start_i + 4][(block_start_j + 4 + 8) % n][1])
            wm_bits.append(1 if delta < -T or (0 < delta < T) else 0)

    wm_matrix = [[0] * 64 for _ in range(64)]
    for i in range(len(wm_bits)):
        wm_matrix[i // 64][i % 64] = wm_bits[i]
    # wm_matrix = arnold_regeneration(wm_matrix)
    wm_matrix_img = [[0] * 64 for _ in range(64)]
    for i in range(len(wm_matrix)):
        for j in range(len(wm_matrix[i])):
            if wm_matrix[i][j] == 1:
                wm_matrix_img[i][j] = [255, 255, 255, 255]
            else:
                wm_matrix_img[i][j] = [0, 0, 0, 255]
    out = np.array(wm_matrix_img).astype("uint8")
    return out


# uint8
# print(get_bin_str_of_wm())

# img = Image.open("JasehOnfroyChild.png")
# img_matrix = np.array(img)
# d = dct(dct(img_matrix, axis=0, norm='ortho'), axis=1, norm='ortho')
# after_shake = idct(idct(d, axis=0, norm='ortho'), axis=1, norm='ortho')
# Image.fromarray(np.array(after_shake).astype("uint8")).save("AferShake.png")

if __name__ == '__main__':
    mode = input("Enter the mode: embed or ex [em/ex]:\n")
    if mode == "em":
        img = Image.open("JasehOnfroyChild.png")
        img_matrix = np.array(img)
        new_img_matrix = insert(img_matrix)
        Image.fromarray(np.array(new_img_matrix).astype("uint8")).save("waterMarkedImage.png")
        print(f'Small m: {count_of_small_m}')
    else:
        img = Image.open("JasehOnfroyChild.png")
        img_matrix = np.array(img)
        wm = extract_new(img_matrix)
        Image.fromarray(wm).save("waterMarkedImage_extracted.png")
