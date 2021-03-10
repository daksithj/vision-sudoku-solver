import cv2
import numpy as np
import random
import pickle as pkl

size = 28

fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN,
         cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_ITALIC]

numbers = []

for a in range(70000):

    x_loc = random.uniform(0.1, 0.5)
    y_loc = random.uniform(0.2, 0.4)

    scale = size / random.uniform(30, 50)

    thickness = random.randint(1, 2)

    text_x = int(size * x_loc)
    text_y = int(size * -0.3) + size

    canvas = np.zeros((size, size, 3))

    number = random.randint(0, 9)
    canvas = cv2.putText(canvas, str(number), (text_x, text_y), random.choice(fonts), scale, (1, 1, 1), thickness)

    canvas = canvas * 255
    canvas = np.asarray(canvas).astype(np.uint8)
    numbers.append((a, number))
    cv2.imwrite(f'ocr_dataset/{a}.png', canvas)

with open('number_list.pkl', 'wb') as f:
    pkl.dump(numbers, f)
