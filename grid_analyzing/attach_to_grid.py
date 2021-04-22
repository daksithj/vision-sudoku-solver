import cv2
import numpy as np
from grid_analyzing.grid_generator import order_points


def unwarp_image(warped_puzzle, original_image, ordered_points, warped_quadrangle_points):
    ordered_points = np.array(ordered_points)

    # Perspective transformation between original plane and warped plane
    homography, _ = cv2.findHomography(warped_quadrangle_points, ordered_points)

    # Un-warp image
    warped = cv2.warpPerspective(warped_puzzle, homography, (original_image.shape[1], original_image.shape[0]))

    # Fill puzzle area in the original image with black
    cv2.fillConvexPoly(original_image, ordered_points, 0, 16)

    # Add the un-warped sudoku puzzle to the black area
    unwarped_image = cv2.add(original_image, warped)

    return unwarped_image


def print_on_screen(warped_puzzle, empty_cells, solution, original_image, puzzle_contour, warped_quadrangle_points):
    # output_image = puzzle.copy()

    for val in empty_cells:
        start_x, start_y, end_x, end_y = val['location']

        # The font size as a scale
        font_scale = (end_x - start_x) / 50

        thickness = (end_x - start_x) // 30

        # bottom left corner coordinate to print number
        text_x = int((end_x - start_x) * 0.33)
        text_y = int((end_y - start_y) * -0.2)
        text_x += start_x
        text_y += end_y

        index = val['index']
        cv2.putText(warped_puzzle, str(solution[index[1]][index[0]]), (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    ordered_points = order_points(puzzle_contour.reshape(4, 2)).astype('int32')

    print_image = unwarp_image(warped_puzzle, original_image, ordered_points, warped_quadrangle_points)

    return print_image
