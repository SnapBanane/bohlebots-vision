import cv2
import numpy as np
import math
import imutils

# Globale Variablen
current_frame_for_callback = None
trackbar_window_name = "Trackbars"
camera_window_name = "Original Kamerafeed - Klicken zum Kalibrieren (Circle Detect)"
mask_window_name = "Farbmaske (bereinigt)"
result_window_name = "Ergebnis (Maskiertes Bild)"
circle_filtered_mask_window_name = "Circle-Filtered Mask"

# --- ROI und Circle Detection Parameter ---
SEARCH_ROI_SIZE = 100  # z.B. 100x100 Pixel Suchbereich
FALLBACK_SQUARE_ROI_SIZE = 25
# Parameter für HoughCircles im Such-ROI (müssen ggf. angepasst werden)
HC_DP = 1.2          # Inverse ratio of accumulator resolution
HC_MIN_DIST = SEARCH_ROI_SIZE // 4 # Mindestabstand zwischen erkannten Kreisen
HC_PARAM1 = 60       # Oberer Canny-Schwellenwert
HC_PARAM2 = 25       # Akkumulator-Schwellenwert (kleiner = mehr Kreise)
HC_MIN_RADIUS_SEARCH = 5  # Minimaler Radius für Kreise im Such-ROI
HC_MAX_RADIUS_SEARCH = SEARCH_ROI_SIZE // 2 # Maximaler Radius

# Globale Variablen für Visualisierung
last_search_roi_rect = None
last_detected_circle_params = None # (center_x_global, center_y_global, radius_global)
last_fallback_roi_rect = None
last_sampled_avg_bgr_display = None

def nothing(x):
    pass

def is_circle(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if 0.7 <= circularity <= 1.2:
        return True
    return False

def distance_pts(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def sample_area_hsv(hsv_frame, center_x, center_y, sample_radius=15):
    h, w = hsv_frame.shape[:2]
    
    # Create circular mask for sampling
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), sample_radius, 255, -1)
    
    # Extract HSV values within the mask
    hsv_values = hsv_frame[mask > 0]
    
    if len(hsv_values) == 0:
        return None, None
    
    # Calculate mean and std for each channel
    mean_hsv = np.mean(hsv_values, axis=0)
    std_hsv = np.std(hsv_values, axis=0)
    
    return mean_hsv, std_hsv

def calculate_adaptive_tolerances(mean_hsv, std_hsv):
    h_mean, s_mean, v_mean = mean_hsv
    h_std, s_std, v_std = std_hsv if std_hsv is not None else (5, 20, 20)
    
    h_base_tolerance = 20
    s_base_tolerance = 80
    v_base_tolerance = 80
    
    h_tolerance = max(h_base_tolerance, int(h_std * 3))
    s_tolerance = max(s_base_tolerance, int(s_std * 2.5))
    v_tolerance = max(v_base_tolerance, int(v_std * 2.5))
    
    if s_mean < 50:
        s_tolerance = min(255, int(s_tolerance * 1.5))
        h_tolerance = min(179, int(h_tolerance * 1.3))
    
    if v_mean < 80:
        v_tolerance = min(255, int(v_tolerance * 1.4))
    
    if h_mean < 15 or h_mean > 165:
        h_tolerance = min(179, int(h_tolerance * 1.2))
    
    h_tolerance = min(h_tolerance, 35)
    s_tolerance = min(s_tolerance, 120)
    v_tolerance = min(v_tolerance, 120)
    
    return h_tolerance, s_tolerance, v_tolerance

def mouse_callback(event, x_click, y_click, flags, param):
    global current_frame_for_callback, trackbar_window_name
    global last_search_roi_rect, last_detected_circle_params, last_fallback_roi_rect, last_sampled_avg_bgr_display

    if event == cv2.EVENT_LBUTTONDOWN and current_frame_for_callback is not None:
        full_frame_hsv = cv2.cvtColor(current_frame_for_callback, cv2.COLOR_BGR2HSV)
        frame_height, frame_width = current_frame_for_callback.shape[:2]

        half_search = SEARCH_ROI_SIZE // 2
        sx1, sy1 = max(0, x_click - half_search), max(0, y_click - half_search)
        sx2, sy2 = min(frame_width, x_click + half_search), min(frame_height, y_click + half_search)
        
        last_search_roi_rect = (sx1, sy1, sx2 - sx1, sy2 - sy1)
        last_detected_circle_params, last_fallback_roi_rect = None, None

        search_roi_bgr = current_frame_for_callback[sy1:sy2, sx1:sx2]
        if search_roi_bgr.size == 0: return

        search_roi_gray = cv2.cvtColor(search_roi_bgr, cv2.COLOR_BGR2GRAY)
        search_roi_blurred = cv2.GaussianBlur(search_roi_gray, (9, 9), 2)
        
        circles_in_roi = cv2.HoughCircles(search_roi_blurred, cv2.HOUGH_GRADIENT,
                                          dp=HC_DP, minDist=HC_MIN_DIST,
                                          param1=HC_PARAM1, param2=HC_PARAM2,
                                          minRadius=HC_MIN_RADIUS_SEARCH, maxRadius=HC_MAX_RADIUS_SEARCH)
        
        best_circle_global = None
        if circles_in_roi is not None:
            circles_in_roi = np.uint16(np.around(circles_in_roi))
            min_dist_to_click = float('inf')
            
            for c_roi in circles_in_roi[0, :]:
                center_x_global, center_y_global = sx1 + c_roi[0], sy1 + c_roi[1]
                dist = distance_pts((x_click, y_click), (center_x_global, center_y_global))
                if dist < min_dist_to_click and c_roi[2] > 0:
                    min_dist_to_click = dist
                    best_circle_global = (center_x_global, center_y_global, c_roi[2])
            
            if best_circle_global:
                last_detected_circle_params = best_circle_global

        avg_hsv_for_calibration, sampled_bgr_for_display = None, None
        std_hsv = None

        if best_circle_global:
            c_x, c_y, r = best_circle_global
            mean_hsv, std_hsv = sample_area_hsv(full_frame_hsv, c_x, c_y, min(r, 20))
            if mean_hsv is not None:
                avg_hsv_for_calibration = mean_hsv
                circle_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cv2.circle(circle_mask, (c_x, c_y), r, 255, -1)
                sampled_bgr_for_display = cv2.mean(current_frame_for_callback, mask=circle_mask)[:3]
            else:
                best_circle_global, last_detected_circle_params = None, None

        if not best_circle_global:
            mean_hsv, std_hsv = sample_area_hsv(full_frame_hsv, x_click, y_click, FALLBACK_SQUARE_ROI_SIZE // 2)
            if mean_hsv is not None:
                avg_hsv_for_calibration = mean_hsv
                half_fallback = FALLBACK_SQUARE_ROI_SIZE // 2
                fx1, fy1 = max(0, x_click - half_fallback), max(0, y_click - half_fallback)
                fx2, fy2 = min(frame_width - 1, x_click + half_fallback), min(frame_height - 1, y_click + half_fallback)
                if fx1 < fx2 and fy1 < fy2:
                    fallback_roi_bgr = current_frame_for_callback[fy1:fy2+1, fx1:fx2+1]
                    sampled_bgr_for_display = np.mean(fallback_roi_bgr, axis=(0,1))
                    last_fallback_roi_rect = (fx1, fy1, (fx2+1)-fx1, (fy2+1)-fy1)

        if avg_hsv_for_calibration is None: return

        last_sampled_avg_bgr_display = tuple(map(int, sampled_bgr_for_display))
        h_avg, s_avg, v_avg = map(int, avg_hsv_for_calibration)

        h_tol, s_tol, v_tol = calculate_adaptive_tolerances(avg_hsv_for_calibration, std_hsv)

        l_h, l_s, l_v = max(0, h_avg - h_tol), max(0, s_avg - s_tol), max(0, v_avg - v_tol)
        u_h, u_s, u_v = min(179, h_avg + h_tol), min(255, s_avg + s_tol), min(255, v_avg + v_tol)
        
        print("-" * 30)
        print(f"Sampled BGR: {last_sampled_avg_bgr_display}")
        print(f"Sampled HSV: ({h_avg}, {s_avg}, {v_avg})")
        print(f"New Lower Bound: [{l_h}, {l_s}, {l_v}]")
        print(f"New Upper Bound: [{u_h}, {u_s}, {u_v}]")
        print("-" * 30)

        cv2.setTrackbarPos("H_min", trackbar_window_name, l_h)
        cv2.setTrackbarPos("H_max", trackbar_window_name, u_h)
        cv2.setTrackbarPos("S_min", trackbar_window_name, l_s)
        cv2.setTrackbarPos("S_max", trackbar_window_name, u_s)
        cv2.setTrackbarPos("V_min", trackbar_window_name, l_v)
        cv2.setTrackbarPos("V_max", trackbar_window_name, u_v)

cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit()

cv2.namedWindow(trackbar_window_name)
cv2.namedWindow(camera_window_name)
cv2.namedWindow(mask_window_name)
cv2.namedWindow(result_window_name)
cv2.namedWindow(circle_filtered_mask_window_name)
cv2.setMouseCallback(camera_window_name, mouse_callback)

cv2.createTrackbar("H_min", trackbar_window_name, 0, 179, nothing)
cv2.createTrackbar("S_min", trackbar_window_name, 0, 255, nothing)
cv2.createTrackbar("V_min", trackbar_window_name, 0, 255, nothing)
cv2.createTrackbar("H_max", trackbar_window_name, 179, 179, nothing)
cv2.createTrackbar("S_max", trackbar_window_name, 255, 255, nothing)
cv2.createTrackbar("V_max", trackbar_window_name, 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret: break
    current_frame_for_callback = frame.copy()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min, s_min, v_min = cv2.getTrackbarPos("H_min", trackbar_window_name), cv2.getTrackbarPos("S_min", trackbar_window_name), cv2.getTrackbarPos("V_min", trackbar_window_name)
    h_max, s_max, v_max = cv2.getTrackbarPos("H_max", trackbar_window_name), cv2.getTrackbarPos("S_max", trackbar_window_name), cv2.getTrackbarPos("V_max", trackbar_window_name)

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    
    kernel = np.ones((5,5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    
    # only show circle filtered masks
    circle_filtered_mask = np.zeros_like(mask_cleaned)
    cnts = cv2.findContours(mask_cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if is_circle(c):
            cv2.drawContours(circle_filtered_mask, [c], -1, 255, -1)

    result_frame = cv2.bitwise_and(frame, frame, mask=mask_cleaned)

    display_frame_feedback = frame.copy()
    if last_search_roi_rect:
        rx, ry, rw, rh = last_search_roi_rect
        cv2.rectangle(display_frame_feedback, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1)
    if last_detected_circle_params:
        cx, cy, r = last_detected_circle_params
        cv2.circle(display_frame_feedback, (cx, cy), r, (0, 255, 0), 2)
    elif last_fallback_roi_rect:
        frx, fry, frw, frh = last_fallback_roi_rect
        cv2.rectangle(display_frame_feedback, (frx, fry), (frx + frw, fry + frh), (0, 0, 255), 1)

    if last_sampled_avg_bgr_display:
        cv2.rectangle(display_frame_feedback, (10, 10), (50, 50), last_sampled_avg_bgr_display, -1)
        cv2.rectangle(display_frame_feedback, (10, 10), (50, 50), (0,0,0), 1)

    lower_text = f"Lower: {lower_bound}"
    upper_text = f"Upper: {upper_bound}"
    cv2.putText(display_frame_feedback, lower_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(display_frame_feedback, lower_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display_frame_feedback, upper_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(display_frame_feedback, upper_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    cv2.imshow(camera_window_name, display_frame_feedback)
    cv2.imshow(mask_window_name, mask_cleaned)
    cv2.imshow(circle_filtered_mask_window_name, circle_filtered_mask)
    cv2.imshow(result_window_name, result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()