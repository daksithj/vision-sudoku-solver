import os
import cv2
import numpy as np
import random
import pickle as pkl
import imutils
from keras.utils import Sequence, np_utils

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


class NumberDataSet(Sequence):

    def __init__(self, data_size,  batch_size, image_size):
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_size = data_size

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):

        x_train = []
        y_train = []

        number_count = [(self.batch_size // 10) for _ in range(10)]

        possible_numbers = [i for i in range(10)]

        count = 0

        while count < self.batch_size:

            num = random.choice(possible_numbers)

            if number_count[num] == 0:
                possible_numbers.remove(num)
                continue

            num_image = self.generate_num(num)

            x_train.append(num_image)
            y_train.append(num)

            count += 1

        x_train = (np.asarray(x_train) / 255.0).astype('float32')
        y_train = (np_utils.to_categorical(y_train)).astype('float32')

        return x_train, y_train

    def generate_num(self, num):

        x_loc = random.uniform(0.1, 0.4)
        y_loc = random.uniform(-0.3, -0.1)

        scale_div = random.uniform(25, 40)

        scale = self.image_size / scale_div

        thickness = random.randint(1, 5)

        text_x = int(self.image_size * x_loc)
        text_y = int(self.image_size * y_loc) + self.image_size

        if scale_div < 30:
            text_x = int(self.image_size * 0.2)
            text_y = int(self.image_size * -0.15) + self.image_size

        canvas = np.zeros((self.image_size, self.image_size, 3))

        canvas = cv2.putText(canvas, str(num), (text_x, text_y), random.choice(fonts), scale, (1, 1, 1), thickness)

        canvas = canvas * 255
        canvas = np.asarray(canvas).astype(np.uint8)

        rotate_chance = random.randint(0, 100)

        if rotate_chance > 75:
            canvas = imutils.rotate(canvas, random.randint(-30, 30))

        aug_chance = random.randint(0, 100)

        if aug_chance < 20:
            canvas = cv2.GaussianBlur(canvas, (3, 3), random.uniform(0.5, 3))

        aug_chance = random.randint(0, 100)

        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        if aug_chance < 20:
            gauss = np.random.normal(0, 0.1, canvas.shape)
            canvas = canvas + gauss

        canvas = np.expand_dims(canvas, axis=2)
        return canvas



if __name__ == '__main__':
    data_set = NumberDataSet(10000, 200, 64)

    data, num = (data_set.__getitem__(0))
    print(data.shape)