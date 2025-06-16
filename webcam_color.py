import cv2
import numpy as np
import math

# Globale Variablen
current_frame_for_callback = None
trackbar_window_name = "Trackbars"
camera_window_name = "Original Kamerafeed - Klicken zum Kalibrieren (Circle Detect)"
mask_window_name = "Farbmaske (bereinigt)"
result_window_name = "Ergebnis (Maskiertes Bild)"

# --- ROI und Circle Detection Parameter ---
# Größe des quadratischen Bereichs um den Klick, in dem nach Kreisen gesucht wird
SEARCH_ROI_SIZE = 100  # z.B. 100x100 Pixel Suchbereich
# Fallback ROI Größe, falls kein Kreis gefunden wird (ähnlich wie vorherige ROI_SIZE)
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

def distance_pts(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def mouse_callback(event, x_click, y_click, flags, param):
    global current_frame_for_callback, trackbar_window_name
    global last_search_roi_rect, last_detected_circle_params, last_fallback_roi_rect, last_sampled_avg_bgr_display

    if event == cv2.EVENT_LBUTTONDOWN and current_frame_for_callback is not None:
        full_frame_hsv = cv2.cvtColor(current_frame_for_callback, cv2.COLOR_BGR2HSV)
        frame_height, frame_width = current_frame_for_callback.shape[:2]

        # 1. Definiere Such-ROI um den Klick
        half_search = SEARCH_ROI_SIZE // 2
        sx1 = max(0, x_click - half_search)
        sy1 = max(0, y_click - half_search)
        sx2 = min(frame_width, x_click + half_search) # Slicing exklusiv, daher frame_width ist ok
        sy2 = min(frame_height, y_click + half_search)
        
        last_search_roi_rect = (sx1, sy1, sx2 - sx1, sy2 - sy1) # Für Visualisierung
        last_detected_circle_params = None # Reset
        last_fallback_roi_rect = None      # Reset

        search_roi_bgr = current_frame_for_callback[sy1:sy2, sx1:sx2]
        if search_roi_bgr.size == 0:
            print("Such-ROI ist leer.")
            return

        # 2. Kreiserkennung im Such-ROI
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
                # Konvertiere Kreis-Koordinaten (relativ zum Such-ROI) zu globalen Koordinaten
                center_x_roi, center_y_roi, radius_roi = c_roi[0], c_roi[1], c_roi[2]
                center_x_global = sx1 + center_x_roi
                center_y_global = sy1 + center_y_roi
                
                dist = distance_pts((x_click, y_click), (center_x_global, center_y_global))
                if dist < min_dist_to_click and radius_roi > 0: # Wähle den Kreis, dessen Zentrum am nächsten zum Klick ist
                    min_dist_to_click = dist
                    best_circle_global = (center_x_global, center_y_global, radius_roi)
            
            if best_circle_global:
                last_detected_circle_params = best_circle_global
                print(f"Kreis gefunden bei ({best_circle_global[0]},{best_circle_global[1]}) mit Radius {best_circle_global[2]}")


        # 3. Farbmittelung
        avg_hsv_for_calibration = None
        sampled_bgr_for_display = None

        if best_circle_global:
            # Mittelung innerhalb des erkannten Kreises
            c_x, c_y, r = best_circle_global
            circle_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            cv2.circle(circle_mask, (c_x, c_y), r, 255, -1)
            
            if np.sum(circle_mask) > 0: # Sicherstellen, dass die Maske nicht leer ist
                avg_hsv_for_calibration = cv2.mean(full_frame_hsv, mask=circle_mask)[:3] # Nur H, S, V
                # Für die Anzeige der gesampelten Farbe, den Durchschnitt der BGR-Werte nehmen
                sampled_bgr_for_display = cv2.mean(current_frame_for_callback, mask=circle_mask)[:3]
            else:
                print("Kreismaske war leer, falle zurück.")
                best_circle_global = None # Erzwinge Fallback
                last_detected_circle_params = None


        if not best_circle_global: # Fallback auf quadratische ROI
            print("Kein passender Kreis gefunden. Nutze quadratische Fallback-ROI.")
            half_fallback = FALLBACK_SQUARE_ROI_SIZE // 2
            fx1 = max(0, x_click - half_fallback)
            fy1 = max(0, y_click - half_fallback)
            fx2 = min(frame_width -1 , x_click + half_fallback) # -1 da exklusiv in slicing
            fy2 = min(frame_height -1, y_click + half_fallback)

            if fx1 >= fx2 or fy1 >= fy2:
                print("Fallback ROI ungültig.")
                return

            fallback_roi_bgr = current_frame_for_callback[fy1:fy2+1, fx1:fx2+1]
            if fallback_roi_bgr.size > 0:
                sampled_bgr_for_display = np.mean(fallback_roi_bgr, axis=(0,1))
                fallback_roi_hsv = cv2.cvtColor(np.uint8([[sampled_bgr_for_display]]), cv2.COLOR_BGR2HSV)[0][0]
                avg_hsv_for_calibration = fallback_roi_hsv
                last_fallback_roi_rect = (fx1, fy1, (fx2+1)-fx1, (fy2+1)-fy1)
            else:
                print("Fallback ROI war leer.")
                return
        
        if avg_hsv_for_calibration is None:
            print("Konnte keine Farbe sampeln.")
            return

        last_sampled_avg_bgr_display = tuple(map(int, sampled_bgr_for_display))

        h_avg, s_avg, v_avg = int(avg_hsv_for_calibration[0]), int(avg_hsv_for_calibration[1]), int(avg_hsv_for_calibration[2])

        # 4. Trackbars setzen
        h_tolerance = 10
        s_tolerance = 70 
        v_tolerance = 70 

        h_min_new = max(0, h_avg - h_tolerance)
        h_max_new = min(179, h_avg + h_tolerance)
        s_min_new = max(0, s_avg - s_tolerance)
        s_max_new = min(255, s_avg + s_tolerance)
        v_min_new = max(0, v_avg - v_tolerance)
        v_max_new = min(255, v_avg + v_tolerance)

        cv2.setTrackbarPos("H_min", trackbar_window_name, h_min_new)
        cv2.setTrackbarPos("H_max", trackbar_window_name, h_max_new)
        cv2.setTrackbarPos("S_min", trackbar_window_name, s_min_new)
        cv2.setTrackbarPos("S_max", trackbar_window_name, s_max_new)
        cv2.setTrackbarPos("V_min", trackbar_window_name, v_min_new)
        cv2.setTrackbarPos("V_max", trackbar_window_name, v_max_new)

        print(f"Gesampelte Avg HSV: [{h_avg}, {s_avg}, {v_avg}]")
        print(f"Trackbars gesetzt auf: H=[{h_min_new}-{h_max_new}], S=[{s_min_new}-{s_max_new}], V=[{v_min_new}-{v_max_new}]")


# --- Webcam Initialisierung & Hauptschleife (bleibt größtenteils gleich) ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Fehler: Webcam konnte nicht geöffnet werden.")
    exit()

cv2.namedWindow(trackbar_window_name)
cv2.namedWindow(camera_window_name)
cv2.namedWindow(mask_window_name)
cv2.namedWindow(result_window_name)
cv2.setMouseCallback(camera_window_name, mouse_callback)

# Trackbars erstellen (wie zuvor)
cv2.createTrackbar("H_min", trackbar_window_name, 0, 179, nothing)
cv2.createTrackbar("S_min", trackbar_window_name, 0, 255, nothing)
cv2.createTrackbar("V_min", trackbar_window_name, 0, 255, nothing)
cv2.createTrackbar("H_max", trackbar_window_name, 179, 179, nothing)
cv2.createTrackbar("S_max", trackbar_window_name, 255, 255, nothing)
cv2.createTrackbar("V_max", trackbar_window_name, 255, 255, nothing)
# Initiale Werte (wie zuvor)
cv2.setTrackbarPos("H_min", trackbar_window_name, 0); cv2.setTrackbarPos("S_min", trackbar_window_name, 50)
cv2.setTrackbarPos("V_min", trackbar_window_name, 50); cv2.setTrackbarPos("H_max", trackbar_window_name, 179)
cv2.setTrackbarPos("S_max", trackbar_window_name, 255); cv2.setTrackbarPos("V_max", trackbar_window_name, 255)

print("Webcam wird initialisiert...")
print(f"Klicken Sie in '{camera_window_name}', um Farbe zu kalibrieren (versucht Kreisdetektion).")
print("Drücken Sie 'q', um zu beenden.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    current_frame_for_callback = frame.copy()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H_min", trackbar_window_name)
    s_min = cv2.getTrackbarPos("S_min", trackbar_window_name)
    v_min = cv2.getTrackbarPos("V_min", trackbar_window_name)
    h_max = cv2.getTrackbarPos("H_max", trackbar_window_name)
    s_max = cv2.getTrackbarPos("S_max", trackbar_window_name)
    v_max = cv2.getTrackbarPos("V_max", trackbar_window_name)

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    kernel = np.ones((5,5),np.uint8)
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    result_frame = cv2.bitwise_and(frame, frame, mask=mask_cleaned)

    # Visualisierungen
    display_frame_feedback = frame.copy()
    if last_search_roi_rect:
        rx, ry, rw, rh = last_search_roi_rect
        cv2.rectangle(display_frame_feedback, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 1) # Cyan Such-ROI
    if last_detected_circle_params:
        cx, cy, r = last_detected_circle_params
        cv2.circle(display_frame_feedback, (cx, cy), r, (0, 255, 0), 2) # Grüner erkannter Kreis
    elif last_fallback_roi_rect: # Nur zeichnen, wenn Fallback genutzt wurde UND kein Kreis
        frx, fry, frw, frh = last_fallback_roi_rect
        cv2.rectangle(display_frame_feedback, (frx, fry), (frx + frw, fry + frh), (0, 0, 255), 1) # Rote Fallback-ROI

    if last_sampled_avg_bgr_display:
        # Zeige Farbfeld oben links oder neben dem Klickpunkt
        swatch_x, swatch_y = 10, 10 
        if last_search_roi_rect: # Positioniere es neben der Such-ROI, wenn möglich
            swatch_x = last_search_roi_rect[0] + last_search_roi_rect[2] + 5
            swatch_y = last_search_roi_rect[1]
            if swatch_x + 50 > display_frame_feedback.shape[1]: # Verhindere Überlauf rechts
                swatch_x = 10
            if swatch_y + 50 > display_frame_feedback.shape[0]: # Verhindere Überlauf unten
                 swatch_y = 10


        cv2.rectangle(display_frame_feedback, (swatch_x, swatch_y), (swatch_x + 40, swatch_y + 40), last_sampled_avg_bgr_display, -1)
        cv2.rectangle(display_frame_feedback, (swatch_x, swatch_y), (swatch_x + 40, swatch_y + 40), (0,0,0), 1)


    cv2.imshow(camera_window_name, display_frame_feedback)
    cv2.imshow(mask_window_name, mask_cleaned)
    cv2.imshow(result_window_name, result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nFinale HSV-Werte:")
        print(f"Lower bound: H={h_min}, S={s_min}, V={v_min}")
        print(f"Upper bound: H={h_max}, S={s_max}, V={v_max}")
        print(f"Für Code:")
        print(f"lower_hsv = np.array([{h_min}, {s_min}, {v_min}])")
        print(f"upper_hsv = np.array([{h_max}, {s_max}, {v_max}])")
        break

cap.release()
cv2.destroyAllWindows()