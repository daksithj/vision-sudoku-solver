import time as t
import numpy as np
import cv2
from keras.models import load_model
from grid_analyzing.grid_generator import get_the_grid
from grid_analyzing.attach_to_grid import print_on_screen
from sudoku_solver.sudoku_solver import Sudoku

MODEL_NAME = 'digit_ocr_model/digit_model.h5'


class VisionSudokuError(Exception):
    def __init__(self, message):
        super(VisionSudokuError, self).__init__(message)


def single_image(image_loc):
    start_time = t.time()

    image = cv2.imread(image_loc)

    num_model = load_model(MODEL_NAME)
    read_time = t.time()

    grid_values = get_the_grid(image, num_model)

    if grid_values is not None:
        # Warped puzzle colour image, puzzle contour, quadrangle points of warped image, Sudoku board, empty cell
        # indices and their locations in the warped image
        warped_puzzle, puzzle_contour, warped_quadrangle_vertices, board, empty_cells = grid_values
    else:
        raise VisionSudokuError('Cannot extract puzzle from the image.')

    process_time = t.time()

    sudoku_board = Sudoku(board.copy(), 9)

    solution = sudoku_board.solve()

    solve_time = t.time()

    if solution is None:
        raise VisionSudokuError("The puzzle was not read properly or it is not solvable.")

    print_image = print_on_screen(warped_puzzle, empty_cells, solution, image, puzzle_contour,
                                  warped_quadrangle_vertices)

    final_time = t.time()

    input_time = round(read_time - start_time, 3)
    extract_time = round(process_time - read_time, 3)
    solving_time = round(solve_time - process_time, 3)
    total_time = round(final_time - start_time, 3)

    print(f'Time to read image: {input_time} seconds')
    print(f'Time pre-process and extract puzzle and digits: {extract_time} seconds')
    print(f'Time to solve the puzzle: {solving_time} seconds')
    print(f'Total time: {total_time} seconds')

    timing = {'input_time': input_time, 'extract_time': extract_time,
              'solving_time': solving_time, 'total_time': total_time}

    return print_image, timing


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

    solution_found = False

    while True:

        if gui.kill_signal:
            break

        time_elapsed = t.time() - prev

        success, img = cap.read()

        gui.update(img)

        if time_elapsed > 1. / frame_rate:
            prev = t.time()

            img_result = img.copy()

            grid_values = get_the_grid(img_result, num_model, True)

            if grid_values is not None:
                # Warped puzzle colour image, puzzle contour, quadrangle points of warped image, Sudoku board,
                # empty cell indices and their locations in the warped image
                warped_puzzle, puzzle_contour, warped_quadrangle_points, board, empty_cells = grid_values
            else:
                continue

            if np.all(board == 0):
                continue

            sudoku_board = Sudoku(board.copy(), 9)
            solution = sudoku_board.solve()

            if solution is None:
                continue

            print_image = print_on_screen(warped_puzzle, empty_cells, solution, img_result, puzzle_contour,
                                          warped_quadrangle_points)
            gui.update(print_image, solution=True)
            solution_found = True
            break

    if not solution_found:
        cap.release()
    cv2.destroyAllWindows()
