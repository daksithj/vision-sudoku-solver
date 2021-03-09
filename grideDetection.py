import cv2
import process_helpers
import numpy as np

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur it
    blur = cv2.GaussianBlur(img_gray, (9, 9), 0)

    # threshold it
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # invert it so the grid lines and text are white
    inverted = cv2.bitwise_not(thresh, 0)

    # get a rectangle kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # morph it to remove some noise like random dots
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)

    # dilate to increase border size
    result = cv2.dilate(morph, kernel, iterations=1)
    return result



def find_contours(img, original):
    # find contours on thresholded image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort by the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = None

    # make sure this is the one we are looking for
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
        num_corners = len(approx)

        if num_corners == 4 and area > 1000:
            polygon = cnt
            break

    if polygon is not None:
        # find its extreme corners
        top_left = process_helpers.find_extreme_corners(polygon, min, np.add)  # has smallest (x + y) value
        top_right = process_helpers.find_extreme_corners(polygon, max, np.subtract)  # has largest (x - y) value
        bot_left = process_helpers.find_extreme_corners(polygon, min, np.subtract)  # has smallest (x - y) value
        bot_right = process_helpers.find_extreme_corners(polygon, max, np.add)  # has largest (x + y) value

        # if its not a square, we don't want it
        if bot_right[1] - top_right[1] == 0:
            return []
        # Checking if horizontal side and vertical side length is approximately equal
        if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
            return []

        cv2.drawContours(original, [polygon], 0, (0, 0, 255), 3)

        # draw corresponding circles
        [process_helpers.draw_extreme_corners(x, original) for x in [top_left, top_right, bot_right, bot_left]]

        return [top_left, top_right, bot_right, bot_left]

    return []


def detectTheGrid(img):
    processed_img = preprocess(img)
    corners = find_contours(processed_img, img.copy())

    if not corners:

        return True
    return False
