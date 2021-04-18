import numpy as np
import imutils
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
from scipy.spatial import distance as dist
import operator

'''
Ordering the coordinate point 
Input : 4 point of the square
Return : Ordered point (Numpy array)
'''


def order_points(points):
    x_sorted = points[np.argsort(points[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]

    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


'''
Take the perspective transform and wrap the grid from the input image 
Input: image (Gray scaled image) , 
       points (4 coordinates of the square)
Return:
      warp: Warped image 
      crop_indices: Transpose coordinates
'''


def get_perspective_transform(image, points):
    rect = order_points(points)

    (tl, tr, br, bl) = rect

    w_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    w_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    max_width = max(int(w_a), int(w_b))

    h_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    h_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(h_a), int(h_b))

    crop_indices = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    perspective_trans = cv2.getPerspectiveTransform(rect, crop_indices)

    warped = cv2.warpPerspective(image, perspective_trans, (max_width, max_height))

    return warped, crop_indices


def find_extreme_corners(polygon, limit_fn, compare_fn):
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    return polygon[section][0][0], polygon[section][0][1]


def draw_extreme_corners(pts, original):
    cv2.circle(original, pts, 7, (0, 255, 0), cv2.FILLED)


'''
Make sure that this is the one we are looking square shape
Input:  puzzle_contour (4 coordinates)
Return: If it is a square return True else return False
'''


def check_contour_format(puzzle_contour):
    top_left = find_extreme_corners(puzzle_contour, min, np.add)
    top_right = find_extreme_corners(puzzle_contour, max, np.subtract)
    bot_right = find_extreme_corners(puzzle_contour, max, np.add)

    if bot_right[1] - top_right[1] == 0:
        return False

    if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
        return False

    return True


'''
Detect the puzzle from the input image
Input : image (image with sudoku grid) 

Output : puzzle_wp_colour - Extracted colored grid 
         puzzle_wp_gray   - Extracted gray colored grid
         puzzle_contour   - Coordinates of the puzzle
         crop_indices     - Coordinates of the perspective transformed puzzle

'''


def find_puzzle(image, is_live_feed=False):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Add gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # Get the threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert colours
    thresh = cv2.bitwise_not(thresh)

    # Get the contours
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Sort contours by perimeter highest to lowest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    puzzle_contour = None

    for c in contours:
        # Get contour perimeter
        peri = cv2.arcLength(c, True)

        area = cv2.contourArea(c)

        # Find an approximate shape (square). The argument is the maximum distance from contour to approximated shape
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Loop done from highest to lowest perimeters. The first 4 sided contour is the puzzle box and area inside
        # the 4 contorts.
        if len(approx) == 4 and area > 1000:
            # The puzzle contour
            puzzle_contour = approx
            break

    if puzzle_contour is None or (is_live_feed and not check_contour_format(puzzle_contour)):
        return None

    # The warped extracted puzzle and its corner coordinates (indexes)
    puzzle_wp_colour, crop_indices = get_perspective_transform(image, puzzle_contour.reshape(4, 2))

    # The warped extracted puzzle (grayscale) and its corner coordinates (indexes)
    puzzle_wp_gray, crop_indices = get_perspective_transform(gray, puzzle_contour.reshape(4, 2))

    return puzzle_wp_colour, puzzle_wp_gray, puzzle_contour, crop_indices


'''
Extract the digit of each cell
Input : cell (single cell of the sudoku puzzle) 

Output : digit - predicted number

'''


def extract_digit(cell):
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) == 0:
        return None

    digit_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")

    cv2.drawContours(mask, [digit_contour], -1, 255, -1)

    (h, w) = thresh.shape
    percent_filled = cv2.countNonZero(mask) / float(w * h)

    if percent_filled < 0.03:
        return None

    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    return digit


def check_allowed_values(allowed_values, x, y, value, box_dim=3):
    # Remove value from allowed list in row with the value
    for idy, allow in enumerate(allowed_values[x]):
        allowed_values[x][idy][value] = False

    # Remove value from allowed list in column with the value
    for idx, allow in enumerate(allowed_values):
        allowed_values[idx][y][value] = False

    begin_row = x - x % box_dim
    begin_column = y - y % box_dim

    # Remove value from allowed list in block with the value
    for i in range(begin_row, begin_row + box_dim):
        for j in range(begin_column, begin_column + box_dim):
            allowed_values[i][j][value] = False


'''
Convert the sudoku puzzle to a two dimesional array
Input : img_result - Image of the sudoku puzzle
        model - Trained OCR model


Output : puzzle_wp_colour  - Extracted colored grid 
         puzzle_contour - Coordinates of the puzzle
         dst - Coordinates of the perspective transformed puzzle
         board - Two dimensional array containing the puzzle values
         print_list - Locations where we need to place numbers on the sudoku puzzle 

'''


def get_the_grid(img_result, model, is_live_feed=False):
    # The warped puzzle,warped puzzle (grayscale), puzzle contour coordinates (in image), warped puzzle corner indices
    found_puzzle = find_puzzle(img_result, is_live_feed)

    if found_puzzle is None:
        return None
    else:
        puzzle_wp_colour, puzzle_wp_gray, puzzle_contour, dst = found_puzzle

    board = np.zeros((9, 9), dtype="int")

    # If puzzle contour was not found
    if puzzle_contour is None:
        return None

    # four_point_transform(img_result.copy(), puzzle_contour.reshape(4, 2))
    print_list = []
    digit_count = 0

    allowed_values = [[[True for _ in range(9)] for _ in range(9)]
                      for _ in range(9)]

    # The length of a cell
    step_x = puzzle_wp_gray.shape[1] // 9
    step_y = puzzle_wp_gray.shape[0] // 9

    # Loop over the grid locations
    for y in range(0, 9):
        for x in range(0, 9):

            # Compute the starting and ending (x, y)-coordinates of the current cell
            start_x = x * step_x
            start_y = y * step_y
            end_x = (x + 1) * step_x
            end_y = (y + 1) * step_y

            # Crop the cell from the warped transform image and then extract the digit from the cell
            cell = puzzle_wp_gray[start_y:end_y, start_x:end_x]
            digit = extract_digit(cell)

            # Verify that the cell is not empty
            if digit is not None:

                # Resize the cell to 28x28 pixels and then prepare the cell for classification
                roi = cv2.resize(digit, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Classify the digit and update the Sudoku board with the prediction
                prediction = model.predict(roi)[0]
                predict_num = np.argsort(-prediction)[:3]
                added_num = False
                for value in predict_num:
                    if value > 0:
                        if allowed_values[y][x][value - 1]:
                            check_allowed_values(allowed_values, y, x, value - 1)
                            board[y, x] = value
                            digit_count += 1
                            added_num = True
                            break
                if not added_num:
                    return None
            else:

                # Coordinates of image to print if it is an empty cell
                print_list.append({"index": (x, y), "location": (start_x, start_y, end_x, end_y)})

    if digit_count < 17:
        return None

    return puzzle_wp_colour, puzzle_contour, dst, board, print_list
