import numpy as np
import imutils
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.segmentation import clear_border
from keras.models import load_model
from scipy.spatial import distance as dist

model = load_model('mnist.h5')


def isASudokupuzzle(c):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    temp = c.copy().reshape(4, 2)
    # Take one corner
    p1 = temp[0]
    countx = 0
    county = 0

    # Check if any length from the selected corner and other corners is less than 50
    for i in range(1, len(temp)):
        if (abs(temp[i][0] - p1[0]) < 50):
            countx += 1
        if (abs(temp[i][1] - p1[1]) < 50):
            county += 1
    if (countx > 1 or county > 1):
        return False

    if len(approx) == 4:
        return True
    return False


def order_points(pts):
    # Take the four coordinates and sort on the x coordinate
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # The leftmost 2 coordinates
    leftMost = xSorted[:2, :]
    # The rightmost 2 coordinates
    rightMost = xSorted[2:, :]

    # Find the top left and bottom left coordinate
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # Find the other corner. Highest euclidean distance from top left corner.
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    # Bottom right and top right corners
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    rect = order_points(pts)
    # Top-left, top-right, bottom-right, bottom-left coordinates
    (tl, tr, br, bl) = rect

    # Find width and heights
    # Think of it has finding hypotenuse length of a triangle

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # We take the max of top width and bottom width to make sure no important part is cropped out
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],                                 # Index for upper left corner
        [maxWidth - 1, 0],                      # Index for upper right corner (-1 because it is an index)
        [maxWidth - 1, maxHeight - 1],          # Index for lower right corner
        [0, maxHeight - 1]], dtype="float32")   # Index for lower left corner

    # Take the corner coordinates and warp them to fit the index above
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped, dst


def find_puzzle(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert colours
    thresh = cv2.bitwise_not(thresh)
    if debug:
        cv2.imshow(" ", thresh)
        cv2.waitKey(0)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # Sort contours by perimeter highest to lowest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    puzzleCnt = None
    for c in cnts:
        # Get contour perimeter
        peri = cv2.arcLength(c, True)
        # Find an approximate shape (square). The argument is the maximum distance from contour to approximated shape
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # Loop done from highest to lowest perimeters. The first 4 sided contour is the puzzle box.
        if len(approx) == 4:
            # The puzzle contour
            puzzleCnt = approx
            break
    if puzzleCnt is None:
        # raise Exception(("Could not find Sudoku puzzle outline. ""Try debugging your thresholding and contour steps."))
        return (None, None, None, None)

    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow(" ", output)
        cv2.waitKey(0)

    # The warped extracted puzzle and its corner coordinates (indexes)
    puzzle, dst = four_point_transform(image, puzzleCnt.reshape(4, 2))
    # The warped extracted puzzle (grayscale) and its corner coordinates (indexes)
    warped, dst = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    if debug:
        cv2.imshow(" ", puzzle)
    return (puzzle, warped, puzzleCnt, dst)


def extract_digit(cell, debug=False):
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return None

    # GEt the contour with the maximum areas
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # Take black image, draw the max contour. 1st arg: black image, 2nd: list of contours (only one here)
    # 3rd: index of contour (-1 means draw all in the list), 4th: Colour 5th: Thickness (-1 means fill it)
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h, w) = thresh.shape
    # See which percentage of image is filled by the above filled contour
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    if percentFilled < 0.03:
        return None
    # Using the above mask using the and operation extract the digit from the image input
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    if debug:
        cv2.imshow(" ", digit)
        cv2.waitKey(0)
    # return the digit to the calling function
    return digit


def getTheGride(img_result):
    # The warped puzzle,warped puzzle (grayscale), puzzle contour coordinates (in image), warped puzzle corner indices
    puzzle, warped, puzzleCnt, dst = find_puzzle(img_result, False)
    board = np.zeros((9, 9), dtype="int")

    # If puzzle contour was not found
    if puzzleCnt is None:
        return puzzle, puzzleCnt, dst, board, None

    # Check if puzzle qualifies for sudoku puzzle (the size of the puzzle)
    if not isASudokupuzzle(puzzleCnt):
        return puzzle, puzzleCnt, dst, board, None

    # out = img.copy()
    # cv2.drawContours(out, [puzzleCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("windowklks", out)
    # cv2.waitKey(0)

    four_point_transform(img_result.copy(), puzzleCnt.reshape(4, 2))
    needToPrint = []

    # The length of a cell
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9
    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cellLocs = []

    digits = []
    # loop over the grid locations
    for y in range(0, 9):
        # initialize the current list of cell locations
        row = []
        for x in range(0, 9):
            # compute the starting and ending (x, y)-coordinates of the
            # current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            # add the (x, y)-coordinates to our cell locations list
            # print(startX, startY)
            row.append((startX, startY, endX, endY))

            # crop the cell from the warped transform image and then
            # extract the digit from the cell
            cell = warped[startY:endY, startX:endX]
            # print("--")
            digit = extract_digit(cell, False)
            digits.append(digit)
            # print(pytesseract.image_to_string(digit))
            # print("as")

            # verify that the digit is not empty
            if digit is not None:
                # resize the cell to 28x28 pixels and then prepare the
                # cell for classification
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                # classify the digit and update the Sudoku board with the
                # prediction
                pred = model.predict(roi).argmax(axis=1)[0]
                # print("--", pred)
                board[y, x] = pred
            else:
                # Coordinates of image to print if it is an empty cell
                needToPrint.append({"index": (x, y), "location": (startX, startY, endX, endY)})

            # add the row to our cell locations
            cellLocs.append(row)

    return puzzle, puzzleCnt, dst, board, needToPrint
