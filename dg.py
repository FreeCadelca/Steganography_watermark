from PIL import Image
import numpy as np
from math import cos
from math import pi
import math


# Best T and K: 140 2

def Cf(i):
    if i == 0:
        return 1 / (2 ** 0.5)
    else:
        return 1


T = 70
K = 10

try_var = 1

def Input():
    string_index = 0
    try:
        source = input("Input image source: ")
        image = Image.open(source)
        try:
            string = input("String message of watermark: ")
            destination = input("Output image destination: ")
            wtr_dest = input("Watermark image destination: ")
            channel = 1
            count = 0
            img_array = np.array(image)
            dcp_array_prepared = [[0] * len(img_array[0]) for _ in range(len(img_array))]
            dcp_array = [[0] * len(img_array[0]) for _ in range(len(img_array))]
            wtrmark = np.zeros((len(img_array) // 8, len(img_array[0]) // 8, 4), dtype=np.uint8)
            for x in range(len(img_array)):
                for y in range(len(img_array[0])):
                    dcp_array_prepared[x][y] = (img_array[x][y][channel] - 128)  # 0 => 128
                    """k = 127
                   if dcp_array_prepared[x][y] < -1 * k:
                       dcp_array_prepared[x][y] = -1 * k
                    if dcp_array_prepared[x][y] > k:
                        dcp_array_prepared[x][y] = k"""
             # print("Arrays prepared")
             # print("Started dcp")
            """"Здесь начало ДКП"""
            for i in range(0, len(dcp_array), 8):
                for j in range(0, len(dcp_array[0]) - 8, 8):
                    for i_i in range(8):
                        for j_i in range(8):
                            c = Cf(i_i) * Cf(j_i) / 4 / try_var
                            for x in range(8):
                                for y in range(8):
                                    dcp_array[i + i_i][j + j_i] += c * cos((2 * x + 1) * i_i * pi / 16) * cos(
                                        (2 * y + 1) * j_i * pi / 16) * dcp_array_prepared[i + x][j + y]
            """"Здесь конец ДКП"""
        # print("Ended dcp")
            for i in range(0, len(dcp_array), 8):
                for j in range(len(dcp_array[0]) - 16, -1, -8):
                    if count < len(string):
                        Med_finder = \
                            [dcp_array[i][j + 3], dcp_array[i][j + 1], dcp_array[i][j + 2], dcp_array[i + 1][j],
                             dcp_array[i + 1][j + 1],
                             dcp_array[i + 1][j + 2], dcp_array[i + 2][j], dcp_array[i + 2][j + 1], dcp_array[i + 3][j]]
                        Med_finder.sort()
                        Med = Med_finder[4]

                        if abs(dcp_array[i][j]) > 1000 or abs(dcp_array[i][j]) < 1:
                            M = 2 * abs(Med)
                        else:
                            M = 2 * abs((dcp_array[i][j] - Med) / dcp_array[i][j])

                        if M < 0.0000001:
                            exponent = math.ceil(math.log10(0.00001 / M))
                            M *= 10 ** exponent

                        # здесь происходит встраивание
                        if string[count] == '1':
                            if dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] > T - K:
                                while dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] > T - K:
                                    dcp_array[i + 3][j + 2] -= M
                            elif K > dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] > -1 * T / 2:
                                while dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] < K:
                                    dcp_array[i + 3][j + 2] += M
                            elif dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] < -1 * T / 2:
                                while dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] > 0 - T - K:
                                    dcp_array[i + 3][j + 2] -= M
                            wtrmark[i // 8][j // 8] = [255, 255, 255, 255]

                        if string[count] == '0':
                            if dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] > T / 2:
                                while dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] <= T + K:
                                    dcp_array[i + 3][j + 2] += M
                            elif -1 * K < dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] < T / 2:
                                while dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] >= -1 * K:
                                    dcp_array[i + 3][j + 2] -= M
                            elif dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] < K - T:
                                while dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] <= K - T:
                                    dcp_array[i + 3][j + 2] += M
                            wtrmark[i // 8][j // 8] = [0, 0, 0, 255]
                        count += 1

            """Здесь ОДКП"""
            for i in range(0, len(dcp_array), 8):
                for j in range(len(dcp_array[0]) - 16, -1, -8):
                    for x in range(8):
                        for y in range(8):
                            dcp_array_prepared[i + x][j + y] = 0
                            for u in range(8):
                                for v in range(8):
                                    dcp_array_prepared[i + x][j + y] += 1 / 4 * try_var * Cf(u) * Cf(v) * dcp_array[i + u][
                                        j + v] * cos(
                                        (2 * x + 1) * u * pi / 16) * cos((2 * y + 1) * v * pi / 16)
                            img_array[i + x][j + y][channel] = (dcp_array_prepared[i + x][j + y] + 128)  # 0 => 128
            out = Image.fromarray(img_array)
            wtrmark_img = Image.fromarray(np.array(wtrmark))
            out.save(destination)
            wtrmark_img.save(wtr_dest)

        except:
            print("Something went wrong")


    except:
        print("No image found")


def Output():
    check = input("Print check (nothing for no check): ")
    try:
        file = input("Input image source: ")
        image = Image.open(file)
        wtr_dest = input("Watermark image destination: ")
        try:
            res = ""
            img_array = np.array(image)
            dcp_array_prepared = [[0] * len(img_array[0]) for _ in range(len(img_array))]
            dcp_array = [[0] * (len(img_array[0])) for _ in range(len(img_array))]
            wtrmark = np.zeros((len(img_array) // 8, len(img_array[0]) // 8, 4), dtype=np.uint8)
            # print("Arrays made")
            for x in range(len(img_array)):
                for y in range(len(img_array[0])):
                    dcp_array_prepared[x][y] = (img_array[x][y][0] - 128)
            for i in range(0, len(dcp_array), 8):
                for j in range(0, len(dcp_array[0]) - 8, 8):
                    for i_i in range(8):
                        for j_i in range(8):
                            c = Cf(i_i) * Cf(j_i) / 4 / try_var
                            for x in range(8):
                                for y in range(8):
                                    dcp_array[i + i_i][j + j_i] += c * cos((2 * x + 1) * i_i * pi / 16) * cos(
                                        (2 * y + 1) * j_i * pi / 16) * dcp_array_prepared[i + x][j + y]
            # print("Ended dcp")
            for i in range(0, len(dcp_array), 8):
                for j in range(len(dcp_array[0]) - 16, -1, -8):
                    if dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] < -1 * T or (
                            dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] > 0 and dcp_array[i + 3][j + 2] -
                            dcp_array[i + 3][j + 10] < T):
                        res += "1"
                        wtrmark[i // 8][j // 8] = [255, 255, 255, 255]
                    elif dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] > T or (
                            dcp_array[i + 3][j + 2] - dcp_array[i + 3][j + 10] < 0 and dcp_array[i + 3][j + 2] -
                            dcp_array[i + 3][j + 10] > -1 * T):
                        res += "0"
                        wtrmark[i // 8][j // 8] = [0, 0, 0, 255]
                    else:
                        res += "x"
                        wtrmark[i // 8][j // 8] = [255, 0, 0, 255]
            if len(check) > 0:
                count = 0
                length = min(len(res), len(check))
                for i in range(length):
                    if res[i] == check[i]:
                        count += 1
                print("BER =", (length - count) / length * 100, "%")
                wtrmark_img = Image.fromarray(np.array(wtrmark))
                wtrmark_img.save(wtr_dest)
        except:
            print("Something went wrong")
    except:
        print("No image found")
    return res

def MSE():
    MSE_index = 0
    try:
        source1 = input("Write image 1 source: ")
        image1 = Image.open(source1)
        source2 = input("Write image 2 source: ")
        image2 = Image.open(source2)
        img1 = np.array(image1)
        img2 = np.array(image2)
    except:
        print("Image not found")
    try:
        for x in range(len(img1)):
            for y in range(len(img1[0])):
                MSE_index += (img1[x][y][0] - img2[x][y][0]) ** 2 /len(img1) / len(img1[0])
        print("MSE =", MSE_index)
    except:
        print("Something went wrong")



if __name__ == '__main__':
    Input()
    print(Output())
    MSE()