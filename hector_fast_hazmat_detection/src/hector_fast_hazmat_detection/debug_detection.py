#!/usr/bin/env python
import argparse
import pickle
import hazmat_detection
import matplotlib.pyplot as plt
import cv2
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug Hazmat Detection')
    parser.add_argument('pickle_path', metavar='PICKLE_PATH', help='Full path of the pickle with the debug information')

    args = parser.parse_args()
    with open(args.pickle_path, 'rb') as handle:
        detection = pickle.load(handle)

    input_image = detection[0]  # type: np.ndarray
    detection_result = detection[1]  # type: hazmat_detection.DetectionResult
    debug_info = detection_result.debug_information

    plt.figure("Input Image")
    plt.imshow(input_image)

    plt.figure("Debug Images")
    plt.subplot(411)
    plt.imshow(detection_result.debug_information.edge_image, "gray")
    plt.subplot(412)
    plt.imshow(detection_result.debug_information.dilated_image, "gray")
    plt.subplot(413)
    out_image = input_image.copy()
    plt.imshow(cv2.drawContours(out_image, debug_info.contours, -1, np.array([255, 0, 0]), 2))
    plt.subplot(414)
    out_image = input_image.copy()
    plt.imshow(cv2.drawContours(out_image, debug_info.filtered_contours, -1, np.array([255, 0, 0]), 2))

    rows = len(debug_info.sub_images)
    figures = (rows + 4) // 5
    for n in range(figures):
        plt.figure("Sub Images %d/%d" % (n+1, figures))
        for i in range(5):
            index = 5 * n + i
            plt.subplot(5, 3, 3 * i + 1)
            plt.imshow(debug_info.sub_images[index][0])
            plt.subplot(5, 3, 3 * i + 2)
            plt.imshow(debug_info.sub_images[index][1], "gray")
            plt.subplot(5, 3, 3 * i + 3)
            out_image = debug_info.sub_images[index][0]
            if debug_info.sub_images[index][2] is not None:
                cv2.drawContours(out_image, [debug_info.sub_images[index][2]], -1, np.array([255, 0, 0]), 2)
            plt.imshow(out_image)

    plt.figure("Detections")
    rows = len(debug_info.detections)
    for i in range(rows):
        plt.subplot(rows, 3, 3 * i + 1)
        plt.imshow(debug_info.detections[i][0])
        plt.subplot(rows, 3, 3 * i + 2)
        plt.imshow(debug_info.detections[i][1])
        plt.subplot(rows, 3, 3 * i + 3)
        plt.title(str(debug_info.detections[i][3]))
        plt.imshow(debug_info.detections[i][2])

    plt.show()

