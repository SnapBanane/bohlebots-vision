import cv2
import numpy as np
import depthai as dai

# Globale Variable, um den aktuellen Frame für den Maus-Callback zu speichern
current_frame_for_callback = None
# Globale Variable, um den Namen des Trackbar-Fensters zu speichern
trackbar_window_name = "Trackbars"
# Globale Variable, um den Namen des Kamerafensters zu speichern
camera_window_name = "Original Kamerafeed"


def nothing(x):
    """Leere Callback-Funktion für Trackbars."""
    pass


def mouse_callback(event, x, y, flags, param):
    """Callback-Funktion für Mausereignisse."""
    global current_frame_for_callback, trackbar_window_name

    if event == cv2.EVENT_LBUTTONDOWN and current_frame_for_callback is not None:
        # BGR-Farbwert des angeklickten Pixels abrufen
        bgr_pixel = current_frame_for_callback[y, x]

        # BGR in HSV konvertieren
        hsv_pixel = cv2.cvtColor(np.uint8([[bgr_pixel]]), cv2.COLOR_BGR2HSV)[0][0]
        h_clicked, s_clicked, v_clicked = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

        # Toleranzen für den HSV-Bereich definieren
        h_tolerance = 10
        s_tolerance = 50
        v_tolerance = 50

        # Neue Min/Max HSV-Werte berechnen
        h_min_new = max(0, h_clicked - h_tolerance)
        h_max_new = min(179, h_clicked + h_tolerance)
        s_min_new = max(0, s_clicked - s_tolerance)
        s_max_new = min(255, s_clicked + s_tolerance)
        v_min_new = max(0, v_clicked - v_tolerance)
        v_max_new = min(255, v_clicked + v_tolerance)

        # Trackbar-Positionen aktualisieren
        cv2.setTrackbarPos("H_min", trackbar_window_name, h_min_new)
        cv2.setTrackbarPos("H_max", trackbar_window_name, h_max_new)
        cv2.setTrackbarPos("S_min", trackbar_window_name, s_min_new)
        cv2.setTrackbarPos("S_max", trackbar_window_name, s_max_new)
        cv2.setTrackbarPos("V_min", trackbar_window_name, v_min_new)
        cv2.setTrackbarPos("V_max", trackbar_window_name, v_max_new)

        print(f"Angeklichter Pixel BGR: {bgr_pixel}, Konvertiert zu HSV: [{h_clicked}, {s_clicked}, {v_clicked}]")
        print(f"Trackbars gesetzt auf: H=[{h_min_new}-{h_max_new}], S=[{s_min_new}-{s_max_new}], V=[{v_min_new}-{v_max_new}]")


# --- OAK Kamera Initialisierung ---
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setPreviewSize(640, 480) # Passen Sie die Auflösung bei Bedarf an
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR) # Wichtig für OpenCV
camRgb.setFps(30)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# Fenster für Trackbars und Kamerafeed erstellen
cv2.namedWindow(trackbar_window_name)
cv2.namedWindow(camera_window_name)

# Maus-Callback für das Kamerafenster setzen
cv2.setMouseCallback(camera_window_name, mouse_callback)

# Trackbars erstellen
cv2.createTrackbar("H_min", trackbar_window_name, 0, 179, nothing)
cv2.createTrackbar("S_min", trackbar_window_name, 0, 255, nothing)
cv2.createTrackbar("V_min", trackbar_window_name, 0, 255, nothing)
cv2.createTrackbar("H_max", trackbar_window_name, 179, 179, nothing)
cv2.createTrackbar("S_max", trackbar_window_name, 255, 255, nothing)
cv2.createTrackbar("V_max", trackbar_window_name, 255, 255, nothing)

print("OAK Kamera wird initialisiert...")
print("Klicken Sie in das Fenster 'Original Kamerafeed', um die HSV-Werte anzupassen.")
print("Passen Sie die HSV-Werte bei Bedarf mit den Schiebereglern weiter an.")
print("Drücken Sie 'q', um das Programm zu beenden und die finalen Werte auszugeben.")

with dai.Device(pipeline) as device:
    print("OAK Kamera gestartet.")
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
        else:
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        if frame is None:
            print("Fehler: Frame konnte nicht von der OAK Kamera gelesen werden.")
            if cv2.waitKey(1) == ord('q'): # Erlaube 'q' zum Beenden auch wenn keine Frames kommen
                break
            continue

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
        result_frame = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow(camera_window_name, frame)
        cv2.imshow("Farbmaske", mask)
        cv2.imshow("Ergebnis (Maskiertes Bild)", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nFinale HSV-Werte:")
            print(f"Lower bound: H={h_min}, S={s_min}, V={v_min}")
            print(f"Upper bound: H={h_max}, S={s_max}, V={v_max}")
            print(f"np.array([{h_min}, {s_min}, {v_min}])")
            print(f"np.array([{h_max}, {s_max}, {v_max}])")
            break

cv2.destroyAllWindows()