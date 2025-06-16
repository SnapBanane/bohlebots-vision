import cv2
import numpy as np
import math
import depthai as dai

# --- OAK Kamera Initialisierung ---
# Pipeline erstellen
pipeline = dai.Pipeline()

# ColorKamera-Knoten erstellen
camRgb = pipeline.create(dai.node.ColorCamera)
# camRgb.setPreviewSize(640, 480) # Alte Vorschaugröße
camRgb.setPreviewSize(1920, 1080)  # Angepasste Vorschaugröße (Full HD als Beispiel)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) # Alte Auflösung
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)  # Wichtig für OpenCV Kompatibilität
camRgb.setFps(30)  # Bei 4K könnten niedrigere FPS stabiler sein, z.B. 30 oder weniger, je nach Modell

# XLinkOut-Knoten für RGB-Frames erstellen
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)  # 'preview' wird hier verwendet, alternativ .video für volle Auflösung

# --- HSV Farbbereiche (AKTUALISIERT) ---
# Tiefblau
lower_deep_blue = np.array([102, 132, 23])
upper_deep_blue = np.array([122, 232, 123])

# Rot (zwei Bereiche wegen des Hue-Wraparounds)
# Erster Bereich aktualisiert, zweiter Bereich für Hue-Wraparound beibehalten
lower_red1 = np.array([0, 165, 90])
upper_red1 = np.array([11, 255, 190])
lower_red2 = np.array([170, 120, 70])  # Ggf. ebenfalls anpassen, falls der Rotbereich über 179 hinausgeht
upper_red2 = np.array([180, 255, 255])

# Tiefes Orange (Ball) - Unverändert, da nicht in der Kalibrierung angegeben
lower_orange = np.array([5, 150, 100])
upper_orange = np.array([20, 255, 255])

# Lila (Magenta)
lower_purple = np.array([157, 145, 31])
upper_purple = np.array([177, 245, 131])

# Cyan/Türkis
lower_cyan = np.array([90, 154, 94])
upper_cyan = np.array([110, 254, 194])

# --- BGR Farben zum Zeichnen ---
COLOR_BLUE_BGR = (255, 0, 0)
COLOR_RED_BGR = (0, 0, 255)
COLOR_PURPLE_BGR = (128, 0, 128)  # BGR für Magenta/Lila
COLOR_CYAN_BGR = (255, 255, 0)
COLOR_ORANGE_BGR = (0, 140, 255)
COLOR_ARROW_BGR = (0, 255, 0)
COLOR_HIGHLIGHT_BASE_BGR = (0, 255, 255)
COLOR_HIGHLIGHT_BALL_BGR = (0, 180, 255)

# Parameter für SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = False
params.filterByArea = True
params.minArea = 80  # Diese Werte müssen ggf. an die höhere Auflösung angepasst werden
params.maxArea = 50000  # Diese Werte müssen ggf. an die höhere Auflösung angepasst werden
params.filterByCircularity = True
params.minCircularity = 0.6
params.filterByConvexity = True
params.minConvexity = 0.80
params.filterByInertia = True
params.minInertiaRatio = 0.3

detector = cv2.SimpleBlobDetector_create(params)

MAX_PAIR_DISTANCE = 150  # Dieser Wert muss ggf. an die höhere Auflösung angepasst werden
MAX_PAIR_DISTANCE_SQ = MAX_PAIR_DISTANCE ** 2


def find_closest_pair(keypoints_base, keypoints_front, max_allowed_sq_dist):
    """Findet das engste Paar von Blobs aus zwei Mengen innerhalb einer maximalen Distanz."""
    best_b_kp = None
    best_f_kp = None
    min_found_sq_distance = float('inf')

    if not keypoints_base or not keypoints_front:
        return None, None

    for b_kp in keypoints_base:
        for f_kp in keypoints_front:
            # Verhindern, dass derselbe Blob mit sich selbst gepaart wird, wenn base und front dieselbe Liste sind
            if b_kp == f_kp and keypoints_base is keypoints_front:  # Identitätsprüfung, nicht nur Gleichheit
                continue
            dx = b_kp.pt[0] - f_kp.pt[0]
            dy = b_kp.pt[1] - f_kp.pt[1]
            sq_dist = dx * dx + dy * dy
            if sq_dist < min_found_sq_distance:
                min_found_sq_distance = sq_dist
                best_b_kp = b_kp
                best_f_kp = f_kp

    if min_found_sq_distance < max_allowed_sq_dist:
        return best_b_kp, best_f_kp
    else:
        return None, None


object_definitions = [
    {"name": "Obj1 (B->R)", "base_color_name": "blue", "front_color_name": "red"},
    {"name": "Obj2 (P->C)", "base_color_name": "purple", "front_color_name": "cyan"},
    {"name": "Obj3 (B->P)", "base_color_name": "blue", "front_color_name": "purple"},
    {"name": "Obj4 (R->C)", "base_color_name": "red", "front_color_name": "cyan"},
]

# --- Hauptschleife mit OAK Kamera ---
with dai.Device(pipeline) as device:
    print("OAK Kamera gestartet...")
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.tryGet()  # Versuche, einen Frame zu bekommen (nicht-blockierend)

        if inRgb is not None:
            # Frame von OAK Kamera abrufen und in OpenCV-Format konvertieren
            frame = inRgb.getCvFrame()
        else:
            # Wenn kein Frame verfügbar ist, kurz warten und nächste Iteration
            if cv2.waitKey(1) == ord('q'):  # Erlaube 'q' zum Beenden auch wenn keine Frames kommen
                break
            continue  # Springe zum nächsten Versuch, einen Frame zu bekommen

        if frame is None:
            print("Error: Could not read frame from OAK camera.")
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_display = frame.copy()

        # --- 1. Masken erstellen ---
        blue_mask = cv2.inRange(hsv_frame, lower_deep_blue, upper_deep_blue)
        red_mask_part1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        red_mask_part2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask_part1, red_mask_part2)
        purple_mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)
        cyan_mask = cv2.inRange(hsv_frame, lower_cyan, upper_cyan)
        orange_mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

        # --- 2. Keypoints (Blobs) erkennen ---
        keypoints_map = {
            "blue": detector.detect(blue_mask),
            "red": detector.detect(red_mask),
            "purple": detector.detect(purple_mask),
            "cyan": detector.detect(cyan_mask),
            "orange": detector.detect(orange_mask)
        }

        # --- 3. Alle erkannten Keypoints zeichnen (optional, zur Fehlersuche) ---
        # for kp_list in keypoints_map.values():
        #     for kp in kp_list:
        #         cv2.circle(frame_display, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2), (255,255,255), 1)

        # --- 4. Datenstrukturen für Ergebnisse initialisieren ---
        detected_objects_info = []

        # --- 5. Objekterkennung basierend auf Farbkombinationen ---
        object_instance_count = 0
        # Kopien der Keypoint-Listen erstellen, um sie modifizieren zu können (falls nötig)
        available_keypoints = {color: list(kps) for color, kps in keypoints_map.items()}

        for i, obj_def in enumerate(object_definitions):
            base_kps_list = available_keypoints[obj_def["base_color_name"]]
            front_kps_list = available_keypoints[obj_def["front_color_name"]]

            best_base_kp, best_front_kp = find_closest_pair(base_kps_list, front_kps_list, MAX_PAIR_DISTANCE_SQ)

            obj_text_line1 = f"{obj_def['name']}: No"
            obj_text_line2 = "Pos: N/A, Angle: N/A"

            if best_base_kp and best_front_kp:
                pt_base = (int(best_base_kp.pt[0]), int(best_base_kp.pt[1]))
                pt_front = (int(best_front_kp.pt[0]), int(best_front_kp.pt[1]))
                position = pt_base
                vector = (pt_front[0] - pt_base[0], pt_front[1] - pt_base[1])

                if vector[0] != 0 or vector[1] != 0:  # Vermeide Division durch Null
                    angle_rad = np.arctan2(vector[1], vector[0])
                    angle_deg = np.degrees(angle_rad)
                    angle_text = f"{angle_deg:.1f} deg"
                else:
                    angle_text = "N/A (same pos)"

                obj_text_line1 = f"{obj_def['name']}: Yes"
                obj_text_line2 = f"Pos: {position}, Angle: {angle_text}"

                cv2.arrowedLine(frame_display, pt_base, pt_front, COLOR_ARROW_BGR, 2)
                cv2.circle(frame_display, pt_base, int(best_base_kp.size / 2) + 3, COLOR_HIGHLIGHT_BASE_BGR, 2)
                cv2.circle(frame_display, pt_front, int(best_front_kp.size / 2) + 2, COLOR_ARROW_BGR,
                           1)  # Highlight front
                object_instance_count += 1

                try:
                    if best_base_kp in base_kps_list:
                        base_kps_list.remove(best_base_kp)
                    if best_front_kp in front_kps_list:
                        front_kps_list.remove(best_front_kp)
                except ValueError:
                    pass

            detected_objects_info.append(obj_text_line1)
            detected_objects_info.append(obj_text_line2)

        # --- 6. Ballerkennung ---
        ball_pos_text = "Ball: N/A"
        if available_keypoints["orange"]:
            sorted_orange_kps = sorted(available_keypoints["orange"], key=lambda k: k.size, reverse=True)
            if sorted_orange_kps:
                largest_orange_kp = sorted_orange_kps[0]
                ball_position = (int(largest_orange_kp.pt[0]), int(largest_orange_kp.pt[1]))
                ball_pos_text = f"Ball Pos: ({ball_position[0]}, {ball_position[1]})"
                cv2.circle(frame_display, ball_position, int(largest_orange_kp.size / 2) + 3, COLOR_HIGHLIGHT_BALL_BGR,
                           2)
        detected_objects_info.append(ball_pos_text)

        # --- 7. Textausgabe auf Frame ---
        y_offset = 20
        for info_line in detected_objects_info:
            cv2.putText(frame_display, info_line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            y_offset += 18

        # --- 8. Bilder anzeigen ---
        cv2.imshow("Camera Feed", frame_display)
        # cv2.imshow("Blue Mask", blue_mask)
        # cv2.imshow("Red Mask", red_mask)
        # cv2.imshow("Purple Mask", purple_mask)
        # cv2.imshow("Cyan Mask", cyan_mask)
        # cv2.imshow("Orange Mask (Ball)", orange_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()