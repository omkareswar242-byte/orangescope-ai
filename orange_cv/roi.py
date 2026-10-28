"""Simple optional rectangle around a centered orange.

This rectangle is based on color, not trained object detection.
"""
import cv2
import numpy as np


def find_fruit_bbox(image):
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 42) & (hsv[:, :, 2] > 48)).astype(np.uint8) * 255

    # Remove little dots and join nearby foreground regions.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    good_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area_ratio = area / (height * width)
        center_x = (x + box_width / 2) / width
        center_y = (y + box_height / 2) / height

        if (0.06 < area_ratio < 0.90 and
                0.15 < center_x < 0.85 and 0.15 < center_y < 0.85 and
                box_width < 0.98 * width and box_height < 0.98 * height):
            good_boxes.append((area, (x, y, box_width, box_height)))

    if not good_boxes:
        return None
    return max(good_boxes, key=lambda item: item[0])[1]
