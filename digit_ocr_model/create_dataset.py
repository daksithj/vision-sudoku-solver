import os
import cv2
import numpy as np
import random
import pickle as pkl
import imutils

size = 28
data_count = 100000
output_dir = "ocr_dataset/"

fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]


def start_generating():

    count = 0

    numbers = []

    data_per_num = int(data_count / 10)

    number_count = [data_per_num for _ in range(10)]

    possible_numbers = [i for i in range(10)]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    while count < data_count:

        num = random.choice(possible_numbers)

        if number_count[num] == 0:
            possible_numbers.remove(num)
            continue

        generate_number(count, num, numbers)
        count += 1

    with open('number_list.pkl', 'wb') as f:
        pkl.dump(numbers, f)


def generate_number(count, num, numbers):
    x_loc = random.uniform(0.1, 0.5)

    scale = size / random.uniform(27, 50)

    thickness = random.randint(1, 2)

    text_x = int(size * x_loc)
    text_y = int(size * -0.3) + size

    canvas = np.zeros((size, size, 3))

    canvas = cv2.putText(canvas, str(num), (text_x, text_y), random.choice(fonts), scale, (1, 1, 1), thickness)

    canvas = canvas * 255
    canvas = np.asarray(canvas).astype(np.uint8)

    rotate_chance = random.randint(0, 100)

    if rotate_chance > 75:

        canvas = imutils.rotate(canvas, random.randint(-30, 30))
    numbers.append((count, num))
    cv2.imwrite(f'{output_dir}{count}.png', canvas)


if __name__ == '__main__':
    start_generating()
