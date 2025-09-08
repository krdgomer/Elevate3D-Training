import cv2
import numpy as np

def post_process(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    biggest = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(biggest, True)
    approx = cv2.approxPolyDP(biggest, epsilon, True)

    clean = np.zeros_like(mask)
    cv2.fillPoly(clean, [approx], 255)
    return clean // 255