import time as t
import numpy as np
import cv2
from keras.models import load_model
from grid_detection import detect_grid
from grid_generator import get_the_grid
from attach_to_grid import print_on_screen
from sudoku_solver import Sudoku


def capture():

    num_model = load_model('digit_model.h5')

    frame_width = 960
    frame_height = 720

    cap = cv2.VideoCapture(0)

    frame_rate = 30

    # Width is id number 3, height is id 4
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    # Change brightness to 150
    cap.set(10, 150)

    prev = 0

    while True:

        time_elapsed = t.time() - prev

        success, img = cap.read()

        if time_elapsed > 1. / frame_rate:
            prev = t.time()

            img_result = img.copy()

            if not detect_grid(img_result):
                continue

            # Warped puzzle, puzzle contour coordinates, warped puzzle index, sudoku puzzle array, printing args
            grid_values = get_the_grid(img_result, num_model)

            if grid_values is not None:
                warped_puzzle, puzzle_contour, crop_indices, board, print_list = grid_values
            else:
                continue

            if np.all(board == 0):
                continue

            sudoku_board = Sudoku(board.copy(), 9)
            solution = sudoku_board.solve()

            if solution is None:
                continue

            print_image = print_on_screen(warped_puzzle, print_list, solution, img_result, puzzle_contour, crop_indices)
            cv2.imshow('window', print_image)
            cv2.waitKey(2000)

        cv2.imshow('window', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    capture()
