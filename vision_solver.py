import time as t
import numpy as np
import cv2
from keras.models import load_model
from grid_detection import detect_grid
from grid_generator import get_the_grid
from attach_to_grid import print_on_screen
from sudoku_solver import Sudoku

MODEL_NAME = 'mnist.h5'


class VisionSudokuError(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        super(VisionSudokuError, self).__init__(message)


def single_image(image_loc):

    image = cv2.imread(image_loc)

    num_model = load_model(MODEL_NAME)

    # if not detect_grid(image):
    #     raise Exception('Cannot detect a grid in the image')

    grid_values = get_the_grid(image, num_model)

    if grid_values is not None:
        warped_puzzle, puzzle_contour, crop_indices, board, print_list = grid_values
    else:
        raise VisionSudokuError('Cannot extract puzzle from the image.')

    sudoku_board = Sudoku(board.copy(), 9)
    solution = sudoku_board.solve()

    if solution is None:
        raise VisionSudokuError("The puzzle was not read properly or it is not solvable.")

    print_image = print_on_screen(warped_puzzle, print_list, solution, image, puzzle_contour, crop_indices)

    return print_image


def initialize_cam():

    frame_width = 960
    frame_height = 720

    cap = cv2.VideoCapture(0)

    frame_rate = 30

    # Width is id number 3, height is id 4
    cap.set(3, frame_width)
    cap.set(4, frame_height)

    # Change brightness to 150
    cap.set(10, 150)
    return cap, frame_rate


def capture(cap, frame_rate, gui):

    num_model = load_model(MODEL_NAME)

    prev = 0

    while True:

        if gui.kill_signal:
            break

        time_elapsed = t.time() - prev

        success, img = cap.read()

        gui.update(img)

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
            gui.update(print_image)
            break

    cap.release()
    cv2.destroyAllWindows()
