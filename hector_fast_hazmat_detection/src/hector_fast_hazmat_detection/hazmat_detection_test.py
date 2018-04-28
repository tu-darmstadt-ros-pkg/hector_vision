#!/usr/bin/env python
import argparse
import hazmat_detection
import cv2
import numpy as np
import os
from timeit import default_timer as timer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hazmat Detection Tester')
    parser.add_argument('template_path', metavar='TEMPLATE_PATH', help='Path to folder with sign templates')
    parser.add_argument('test_dir_path', metavar='TEST_DIRECTORY_PATH',
                        help='Path to folder with test images and ground truths')

    args = parser.parse_args()
    files = sorted(os.listdir(args.test_dir_path))
    test_files = []
    for i in range(len(files) // 2):
        test_files.append((files[2*i], files[2*i+1]))
    detector = hazmat_detection.HazmatSignDetector(args.template_path)

    time = 0
    worst = 0
    for test_pair in test_files:
        image = cv2.imread(os.path.join(args.test_dir_path, test_pair[0]))
        ground_truth_path = os.path.join(args.test_dir_path, test_pair[1])
        if os.path.isfile(ground_truth_path):
            ground_truth = np.genfromtxt(ground_truth_path, delimiter=',', dtype=None,
                                         names=('x', 'y', 'w', 'h', 'name'))
        else:
            ground_truth = None
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: np.ndarray
        start = timer()
        detection_result = detector.detect(input_image)  # type: hazmat_detection.DetectionResult

        check = np.zeros(ground_truth.size if ground_truth is not None else 0, dtype=np.bool)
        print(test_pair[0])
        for result in detection_result.detections:  # type: hazmat_detection.Detection
            match = False
            for i in range(ground_truth.size):
                truth = ground_truth.take(i)
                (x, y, w, h) = cv2.boundingRect(result.contour)
                if truth['name'] == result.name and (x < truth['x'] + truth['w'] and  x + w > truth['x'] and
                y < truth['y'] + truth['h'] and y + h > truth['y']):
                    match = True
                    check[i] = True
                    break
            if not match:
                print("Detected: %s at (%d, %d, %d, %d) which is not in ground truth" % (result.name, x, y, w, h))
                print()
        if check.all():
            print("All signs from ground truth detected! :)")
        else:
            print("These signs weren't detected :(")
            for i in range(len(check)):
                if check[i]:
                    continue
                print(ground_truth.take(i))

        print("-" * 20)
        end = timer()
        duration = end - start
        if duration > worst:
            worst = duration
        time += duration

    print("Took on average %.3f ms" % ((time / len(test_files)) * 1000))
    print("Worst: %.3f ms" % (worst * 1000))

