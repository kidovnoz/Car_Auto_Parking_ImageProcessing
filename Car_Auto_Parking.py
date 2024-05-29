import cv2
import numpy as np

# Load the pre-trained car cascade classifier
car_cascade = cv2.CascadeClassifier('T:\\My Work\\sideview_cascade_classifier.xml')

# Load video
cap = cv2.VideoCapture("T:\\My Work\\parking_map20.mp4")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object to save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('T:\\My Work\\output_parking_detection.avi', fourcc, fps, (frame_width, frame_height))

# Scaling factor (cm per pixel), adjust based on your camera calibration
scaling_factor = 0.1

def process_frame(frame):
    frame_height, frame_width = frame.shape[:2]
    roi_top = frame_height * 2 // 60
    roi_bottom = frame_height
    roi = frame[roi_top:roi_bottom, :]

    cv2.rectangle(frame, (0, roi_top), (frame_width, roi_bottom), (0, 255, 0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    equalized_gray = cv2.equalizeHist(blurred)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    closing_gray = cv2.morphologyEx(equalized_gray, cv2.MORPH_CLOSE, kernel)

    cars = car_cascade.detectMultiScale(closing_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), maxSize=(300, 300))

    _, thresholded = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY_INV)
    closing_thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parking_spots = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5000 < area < 50000:
            x, y, w, h = cv2.boundingRect(contour)
            parking_spots.append((x, y, w, h))

    parking_spots_filtered = []
    for (x, y, w, h) in parking_spots:
        car_found = False
        for (cx, cy, cw, ch) in cars:
            if x < cx + cw and cx < x + w and y < cy + ch and cy < y + h:
                car_found = True
                break
        if not car_found:
            parking_spots_filtered.append((x, y, w, h))

    mid_bottom_x = frame_width // 2
    mid_bottom_y = roi_bottom - roi_top

    for (x, y, w, h) in parking_spots_filtered:
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 255), 2)

        parking_center_x = x + w // 2
        parking_center_y = y + h // 2
        pixel_distance = np.sqrt((mid_bottom_x - parking_center_x) ** 2 + (mid_bottom_y - parking_center_y) ** 2)
        real_distance = pixel_distance * scaling_factor

        cv2.putText(roi, f'{real_distance:.2f} cm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        top_left_x = x
        top_left_y = y
        bottom_right_x = x + w
        bottom_right_y = y + h

        vector_top_left = np.array([top_left_x - mid_bottom_x, top_left_y - mid_bottom_y])
        vector_bottom_right = np.array([bottom_right_x - mid_bottom_x, bottom_right_y - mid_bottom_y])

        dot_product = np.dot(vector_top_left, vector_bottom_right)
        norm_top_left = np.linalg.norm(vector_top_left)
        norm_bottom_right = np.linalg.norm(vector_bottom_right)
        cos_theta = dot_product / (norm_top_left * norm_bottom_right)
        angle_between_vectors = np.arccos(cos_theta) * (180.0 / np.pi)

        cv2.putText(roi, f'{int(angle_between_vectors)} deg', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.line(roi, (mid_bottom_x, mid_bottom_y), (top_left_x, top_left_y), (255, 100, 0), 1)
        cv2.line(roi, (mid_bottom_x, mid_bottom_y), (bottom_right_x, bottom_right_y), (255, 100, 0), 1)

    for (x, y, w, h) in cars:
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 100, 255), 2)

    frame[roi_top:roi_bottom, :] = roi
    out.write(frame)
    cv2.imshow('Parking Detection', frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    process_frame(frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
