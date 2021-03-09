import cv2
import numpy as np
from scipy.spatial import distance as dist


def unwarp_image(img_src, img_dest, pts, time, dst):
    # pts = sorted(pts, key=cv2.contourArea, reverse=True)
    pts = np.array(pts)

    height, width = img_src.shape[0], img_src.shape[1]
    # pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]],
    #                       dtype='float32')
    pts_source = dst
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    # cv2_imshow(warped)
    # In the output image colour the sudoku puzzle area black
    cv2.fillConvexPoly(img_dest, pts, 0, 16)

    # Add the un-warped sudoku puzzle to the black area
    dst_img = cv2.add(img_dest, warped)

    dst_img_height, dst_img_width = dst_img.shape[0], dst_img.shape[1]
    # cv2.putText(dst_img, time, (dst_img_width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    return dst_img


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def printOnTheScrene(puzzle, needToPrint, solvedBord, img_result, puzzleCnt, dst):
    OutputImage = puzzle.copy()

    for val in needToPrint:
        startX, startY, endX, endY = val['location']

        # The font size as a scale
        sclar = (endX - startX) / 50
        # bottom left corner coordinate to print number
        textX = int((endX - startX) * 0.33)
        textY = int((endY - startY) * -0.2)
        textX += startX
        textY += endY

        index = val['index']
        cv2.putText(OutputImage, str(solvedBord[index[1]][index[0]]), (textX, textY),
                    cv2.FONT_HERSHEY_SIMPLEX, sclar, (0, 0, 255), 2)

    outputOne = img_result.copy()
    wr = OutputImage.copy()
    ssss = unwarp_image(wr, outputOne, order_points(puzzleCnt.reshape(4, 2)).astype('int32'), "aa", dst)

    cv2.imshow('window', ssss)
    cv2.waitKey(2000)