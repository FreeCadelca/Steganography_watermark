import math

from PIL import Image
import numpy as np
from scipy.fft import dct, idct
from skimage.metrics import structural_similarity as ssim

T = 80
K = 12


def arnold_transformation(matrix) -> np.array:
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
            old_cords = np.matmul(np.array([[2, -1], [-1, 1]]), np.array([[i], [j]]))
            i_old, j_old = old_cords[0][0] % n, old_cords[1][0] % n
            old_matrix[i_old][j_old] = matrix[i][j]
    return old_matrix


def apply_arnold(wm_bits, is_regeneration=False):
    size = int(np.sqrt(len(wm_bits)))
    wm_bits_2d = np.reshape(wm_bits, (size, size))
    transformed_2d = arnold_regeneration(wm_bits_2d) if is_regeneration else arnold_transformation(wm_bits_2d)
    transformed_1d = transformed_2d.flatten()
    return transformed_1d


def get_bin_str_of_wm(path):
    wm_image = Image.open(path)
    wm_matrix = np.array(wm_image)
    wm_arr = []
    for i in range(len(wm_matrix)):
        for j in range(len(wm_matrix[i])):
            wm_arr.append('1' if wm_matrix[i][j][0] > 127 else '0')
    return ''.join(wm_arr)


def modification(block):
    Z = 2
    DC = block[0][0]
    median_ac = np.median([block[0][1], block[0][2], block[0][3], block[1][0],
                           block[1][1], block[1][2], block[2][0], block[2][1], block[3][0]])
    if abs(DC) > 1000 or abs(DC) < 1:
        M = abs(Z * median_ac)
    else:
        M = abs(Z * (DC - median_ac) / DC)
    if M < 0.00001:
        M = 0.001
    return M


def embed(matrix: np.array, wm_bits):
    wm_bits = ''.join(map(str, apply_arnold(list(map(int, wm_bits)))))

    old_matrix = matrix.copy()
    width, height = matrix.shape[1], matrix.shape[0]
    num_blocks_x = width // 8
    num_blocks_y = height // 8
    index_of_px_in_wm = 0
    for block_i in range(num_blocks_y):
        for block_j in range(num_blocks_x):
            block = [[matrix[block_i * 8 + i][block_j * 8 + j][1] - 128 for j in range(8)] for i in range(8)]
            block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            M = modification(block_dct)
            if block_j < num_blocks_x - 1:
                neighbor_block = [[matrix[block_i * 8 + i][block_j * 8 + j + 8][1] - 128 for j in range(8)]
                                  for i in range(8)]
                neighbor_block_dct = dct(dct(neighbor_block, axis=0, norm='ortho'), axis=1, norm='ortho')
                if index_of_px_in_wm < len(wm_bits):
                    watermark_bit = int(wm_bits[index_of_px_in_wm])
                    delta = block_dct[4][4] - neighbor_block_dct[4][4]
                    if watermark_bit == 1:
                        if delta > T - K:
                            while delta > T - K:
                                block_dct[4][4] -= M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                        elif K > delta > -T / 2:
                            while delta < K:
                                block_dct[4][4] += M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                        elif delta < -T / 2:
                            while delta > -T - K:
                                block_dct[4][4] -= M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                    else:
                        if delta > T / 2:
                            while delta <= T + K:
                                block_dct[4][4] += M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                        elif -K < delta < T / 2:
                            while delta >= -K:
                                block_dct[4][4] -= M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                        elif delta < K - T:
                            while delta <= K - T:
                                block_dct[4][4] += M
                                delta = block_dct[4][4] - neighbor_block_dct[4][4]
                    index_of_px_in_wm += 1
                block_idct = idct(idct(block_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
                for i in range(8):
                    for j in range(8):
                        g = max(min(int(block_idct[i][j]) + 128, 255), 0)
                        matrix[block_i * 8 + i][block_j * 8 + j][1] = g

    print(f"PSNR: {calculate_psnr(old_matrix, matrix)}")
    print(f"MSE: {calculate_mse(old_matrix, matrix)}")
    print(f"RMSE: {math.sqrt(calculate_mse(old_matrix, matrix))}")
    print(f"SSIM: {calculate_ssim(old_matrix, matrix)}")

    return matrix


def calculate_psnr(old_matrix, new_matrix):
    mse = np.mean((old_matrix - new_matrix) ** 2)
    return 9999999 if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))


def calculate_mse(old_matrix, new_matrix):
    return np.mean((old_matrix - new_matrix) ** 2)


def calculate_ssim(old_matrix, new_matrix):
    old_gray = np.array(Image.fromarray(old_matrix).convert('L'), dtype=np.uint8)
    new_gray = np.array(Image.fromarray(new_matrix).convert('L'), dtype=np.uint8)
    return ssim(old_gray, new_gray, data_range=new_gray.max() - new_gray.min())


def extract(matrix, path_of_old_wm):
    width, height = matrix.shape[1], matrix.shape[0]
    num_blocks_x = width // 8
    num_blocks_y = height // 8
    wm_bits = []
    for block_i in range(num_blocks_y):
        for block_j in range(num_blocks_x - 1):
            block = [[matrix[block_i * 8 + i][block_j * 8 + j][1] - 128 for j in range(8)] for i in range(8)]
            block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            delta = block_dct[4][4]
            wm_bits.append(1 if delta < -T or (0 < delta < T) else 0)
    wm_bits += [1] * num_blocks_x
    wm_bits = apply_arnold(wm_bits, is_regeneration=True)

    # Creating matrix of wm
    wm_matrix = [[0] * 64 for _ in range(64)]
    for i in range(len(wm_bits)):
        wm_matrix[i // 64][i % 64] = wm_bits[i]
    wm_matrix_img = [[0] * 64 for _ in range(64)]
    for i in range(len(wm_matrix)):
        for j in range(len(wm_matrix[i])):
            if wm_matrix[i][j] == 1:
                wm_matrix_img[i][j] = [255, 255, 255, 255]
            else:
                wm_matrix_img[i][j] = [0, 0, 0, 255]
    out = np.array(wm_matrix_img).astype("uint8")

    print(f'BER: {calculate_ber(np.array(Image.open(path_of_old_wm)), out)}')
    return out


def calculate_ber(old_matrix, new_matrix):
    old_width, old_height = old_matrix.shape[1], old_matrix.shape[0]
    count_of_right_bits = 0
    for i in range(old_height):
        for j in range(old_width):
            try:
                if (old_matrix[i][j][1] > 127 and new_matrix[i][j][1] > 127 or
                        old_matrix[i][j][1] < 127 and new_matrix[i][j][1] < 127):
                    count_of_right_bits += 1
            except IndexError:
                pass
    return (old_width * old_height - count_of_right_bits) / (old_width * old_height)


if __name__ == '__main__':
    mode = input("Enter the mode: embed or ex [em/ex]:\n")
    if mode == "em":
        path_img = input("Enter name of image-container: ")
        img = Image.open(path_img)
        path_wm = input("Enter name of image-watermark: ")

        img_matrix = np.array(img)
        new_img_matrix = embed(img_matrix, get_bin_str_of_wm(path_wm))
        Image.fromarray(np.array(new_img_matrix).astype("uint8")).save("waterMarkedImage.png")
    else:
        path_img = input("Enter name of image with watermark: ")
        img = Image.open(path_img)
        path_of_old_wm = input("Enter name of source image-watermark (for BER): ")

        img_matrix = np.array(img)
        wm = extract(img_matrix, path_of_old_wm)
        Image.fromarray(wm).save("waterMarkedImage_extracted.png")
# aga.png
# JasehOnfroyChild.png
# watermark.png
# waterMarkedImage.png
# waterMarkedImage_shrinked_quality20.jpg
# waterMarkedImage_shrinked_quality80.jpg
# waterMarkedImage_shrinked_quality100.jpg
# waterMarkedImage_lighted_up.png
# waterMarkedImage_contrast_up.png
# waterMarkedImage_contrast_down.png
# waterMarkedImage_hyper_filtered.png

