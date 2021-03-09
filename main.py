import numpy as np
import cv2
import time as t
from grideDetection import *
from gridGenerator import *
from attacedToGride import printOnTheScrene
from sudokuSolver3by3 import sudokuSolver3by3


def capture():
    frameWidth = 960
    frameHeight = 720
    cap = cv2.VideoCapture(0)
    frame_rate = 30
    # width is id number 3, height is id 4
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    # change brightness to 150
    cap.set(10, 150)
    # load the model with weights
    prev = 0

    while True:

        time_elapsed = t.time() - prev

        success, img = cap.read()

        if time_elapsed > 1. / frame_rate:
            prev = t.time()

            img_result = img.copy()

            if detectTheGrid(img_result):
                continue

            # warped puzzle, puzzle contour coordinates, warped puzzle index, sudoku puzzle array, printing args
            puzzle, puzzleCnt, dst, board, needToPrint = getTheGride(img_result)

            if np.all(board == 0):
                continue

            s = sudokuSolver3by3(9, board.copy())
            solved_bord = s.solve()

            if np.any(solved_bord == 0):
                continue

            printOnTheScrene(puzzle, needToPrint, solved_bord, img_result, puzzleCnt, dst)

        cv2.imshow('window', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    # print_hi('PyCharm')
    capture()
