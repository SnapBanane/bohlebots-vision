import cv2
import numpy as np
import math

# --- Configuration ---
DRAW_COLORS = {
    "blue": (255, 123, 0),
    "yellow": (59, 235, 255),
    "pink": (129, 64, 255)
}

ROBOT_PATTERNS_DEF = {
    "Robot 1": ("blue", "yellow", "pink", "blue"),
    "Robot 2": ("yellow", "pink", "blue", "yellow"),
    "Robot 3": ("pink", "blue", "yellow", "pink"),
    "Robot 4": ("blue", "yellow", "blue", "pink")
}

# --- HSV Color Ranges (FROM YOUR CALIBRATION) ---
HSV_RANGES = {
    "blue":   (np.array([97, 53, 50]), np.array([117, 193, 190])),  # Updated Blue
    "yellow": (np.array([11, 55, 91]), np.array([31, 195, 231])),  # Updated Yellow
    "pink":   (np.array([153, 26, 74]), np.array([173, 166, 214]))  # Previous Pink - to be tested with new method
    # For very reddish pink, you might still need to check ranges near H=0-10 and H=170-179
    # e.g., "pink_low_H": (np.array([0, 50, 50]), np.array([10, 255, 255])),
    #       "pink_high_H": (np.array([170, 50, 50]), np.array([179, 255, 255])),
}


# --- Detection Parameters ---
MIN_CIRCLE_RADIUS_PX = 10 # Min radius for HoughCircles
MAX_CIRCLE_RADIUS_PX = 75 # Max radius for HoughCircles (adjust based on expected size on screen)
HOUGH_PARAM1 = 50   # Upper threshold for Canny edge detector in HoughCircles
HOUGH_PARAM2 = 30   # Accumulator threshold for circle centers in HoughCircles (lower means more circles)
HOUGH_MIN_DIST_BETWEEN_CIRCLES = 20 # Minimum distance between centers of detected circles

# Parameters for identify_robot_patterns (can remain similar)
# ... (keep existing MIN_CIRCLE_AREA, MAX_CIRCLE_AREA, MIN_CIRCULARITY if you revert to contour method)
# ... (keep existing EXPECTED_..._DIST_PX and DIST_TOLERANCE_RATIO)


def classify_circle_color(hsv_frame, center, radius):
    """
    Calculates the average HSV color within a circular region and classifies it.
    Returns the color name (str) or None if not classified.
    """
    mask = np.zeros(hsv_frame.shape[:2], dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    
    # Ensure we don't have an empty mask if circle is off-image or too small
    if np.sum(mask) == 0:
        return None

    mean_hsv = cv2.mean(hsv_frame, mask=mask)
    # mean_hsv will have 4 values (H,S,V,Alpha if present), we only need first 3
    h, s, v = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

    # Check against defined HSV ranges
    for color_name, (lower, upper) in HSV_RANGES.items():
        if lower[0] <= h <= upper[0] and \
           lower[1] <= s <= upper[1] and \
           lower[2] <= v <= upper[2]:
            return color_name
    
    # Special handling for reddish pink if it wraps around H=0/180
    # Example:
    # if (0 <= h <= 10 or 170 <= h <= 179) and \
    #    HSV_RANGES["pink"][0][1] <= s <= HSV_RANGES["pink"][1][1] and \
    #    HSV_RANGES["pink"][0][2] <= v <= HSV_RANGES["pink"][1][2]:
    #     # This logic assumes your "pink" range in HSV_RANGES is for the S and V components
    #     # and you are explicitly checking the H for red here.
    #     # You might need dedicated "pink_low_H" and "pink_high_H" ranges for this.
    #     return "pink" 
        
    return None


def find_and_classify_circles(frame, hsv_frame):
    """
    Detects circles using Hough Transform, then classifies their color by averaging.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Adjust dp, minDist, param1, param2, minRadius, maxRadius as needed
    detected_circles_hough = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of accumulator resolution to image resolution.
        minDist=HOUGH_MIN_DIST_BETWEEN_CIRCLES, # Min distance between centers of detected circles.
        param1=HOUGH_PARAM1, # Upper threshold for the Canny edge detector.
        param2=HOUGH_PARAM2, # Accumulator threshold for circle centers at detection stage.
        minRadius=MIN_CIRCLE_RADIUS_PX,
        maxRadius=MAX_CIRCLE_RADIUS_PX
    )

    classified_circles_list = []
    if detected_circles_hough is not None:
        detected_circles_hough = np.uint16(np.around(detected_circles_hough))
        for pt in detected_circles_hough[0, :]:
            center_x, center_y, radius = pt[0], pt[1], pt[2]
            
            # Ensure radius is not zero to avoid division by zero or empty mask
            if radius == 0:
                continue

            color_name = classify_circle_color(hsv_frame, (center_x, center_y), radius)
            if color_name:
                classified_circles_list.append({
                    "center": (center_x, center_y),
                    "radius": radius,
                    "color": color_name
                    # "contour" is not directly available from HoughCircles,
                    # but not strictly needed if center/radius is sufficient.
                })
    return classified_circles_list

# Keep distance, get_midpoint, and identify_robot_patterns functions as they are.
# The old find_circles_for_color is no longer needed.

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_midpoint(p1, p2):
    return ( (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 )

def identify_robot_patterns(all_detected_circles, frame_to_draw_on):
    """
    Identifies 2x2 robot patterns from a list of all detected circles.
    Returns a list of identified robots, each with its name, center, and orientation vector.
    (This function can largely remain the same, as it expects a list of circle dicts)
    """
    if len(all_detected_circles) < 4:
        return []

    identified_robots = []
    from itertools import combinations

    # Filter out circles that might be too close to be part of different patterns
    # This is a simple way to avoid using the same circle in multiple combinations if they are very close.
    # More sophisticated non-maximum suppression could be used.
    
    # --- Start of existing identify_robot_patterns logic ---
    potential_groups = combinations(all_detected_circles, 4)

    for group_of_4 in potential_groups:
        # Sort by y then x to get a consistent order: TL, TR, BL, BR
        sorted_group = sorted(group_of_4, key=lambda c: (c["center"][1], c["center"][0]))
        
        # Basic geometric check (very loose, can be improved)
        # Cast coordinates to int before subtraction to prevent overflow
        y_coords = [int(c["center"][1]) for c in sorted_group]
        x_coords = [int(c["center"][0]) for c in sorted_group]
        # avg_radius is likely float already due to division, or can be cast if needed
        avg_radius = sum(c['radius'] for c in sorted_group) / 4.0


        # Check if y-coords suggest two rows and x-coords suggest two columns
        # This is a heuristic and might need tuning or a more robust geometric check
        if not (abs(y_coords[0] - y_coords[1]) < avg_radius * 1.5 and \
                abs(y_coords[2] - y_coords[3]) < avg_radius * 1.5 and \
                abs(x_coords[0] - x_coords[2]) < avg_radius * 1.5 and \
                abs(x_coords[1] - x_coords[3]) < avg_radius * 1.5 and \
                (y_coords[2] - y_coords[0]) > avg_radius * 0.5 and \
                (x_coords[1] - x_coords[0]) > avg_radius * 0.5 ): # Ensure some separation
            continue
            
        current_pattern_colors = (
            sorted_group[0]["color"], 
            sorted_group[1]["color"], 
            sorted_group[2]["color"], 
            sorted_group[3]["color"]
        )

        for robot_name, defined_pattern_colors in ROBOT_PATTERNS_DEF.items():
            if current_pattern_colors == defined_pattern_colors:
                all_x_coords_group = [c["center"][0] for c in sorted_group]
                all_y_coords_group = [c["center"][1] for c in sorted_group]
                square_center_x = int(np.mean(all_x_coords_group))
                square_center_y = int(np.mean(all_y_coords_group))
                square_center = (square_center_x, square_center_y)

                p_tl = sorted_group[0]["center"]
                p_tr = sorted_group[1]["center"]
                front_midpoint = get_midpoint(p_tl, p_tr)
                
                min_x = min(all_x_coords_group) - int(avg_radius)
                max_x = max(all_x_coords_group) + int(avg_radius)
                min_y = min(all_y_coords_group) - int(avg_radius)
                max_y = max(all_y_coords_group) + int(avg_radius)
                bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

                # Check if this robot (based on bbox) is already largely identified
                # to prevent multiple detections of the same physical pattern
                is_new_robot = True
                for existing_robot in identified_robots:
                    # Simple IoU (Intersection over Union) or center distance check
                    ex_bbox = existing_robot["bbox"]
                    # Check for significant overlap (e.g., if centers are too close)
                    dist_centers = distance(square_center, existing_robot["center"])
                    if dist_centers < avg_radius * 2: # If centers are closer than twice avg_radius
                        is_new_robot = False
                        break
                if not is_new_robot:
                    continue


                identified_robots.append({
                    "name": robot_name,
                    "center": square_center,
                    "orientation_vector_start": square_center,
                    "orientation_vector_end": front_midpoint,
                    "circles_in_pattern": sorted_group, 
                    "bbox": bbox
                })
                
                cv2.rectangle(frame_to_draw_on, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0), 2)
                cv2.putText(frame_to_draw_on, robot_name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                cv2.arrowedLine(frame_to_draw_on, square_center, orientation_vector_end, (255,0,0), 2, tipLength=0.2)
                
                for circle_data in sorted_group:
                    # Use DRAW_COLORS for visualization
                    cv2.circle(frame_to_draw_on, circle_data["center"], circle_data["radius"], DRAW_COLORS.get(circle_data["color"], (128,128,128)), -1)
                    cv2.circle(frame_to_draw_on, circle_data["center"], circle_data["radius"], (0,0,0), 1)
                
                break 
    return identified_robots
    # --- End of existing identify_robot_patterns logic ---


def main():
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # cv2.namedWindow("Masks", cv2.WINDOW_NORMAL) # Not used with Hough method directly

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # New method: find all potential circles and classify their color
        all_classified_circles = find_and_classify_circles(frame, hsv_frame)
        
        # --- For debugging: Draw all classified circles ---
        # debug_frame_all_circles = frame.copy()
        # for circle_info in all_classified_circles:
        #    cv2.circle(debug_frame_all_circles, circle_info["center"], circle_info["radius"], DRAW_COLORS.get(circle_info["color"], (128,128,128)), 2)
        #    cv2.putText(debug_frame_all_circles, circle_info["color"], (circle_info["center"][0]+10, circle_info["center"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DRAW_COLORS.get(circle_info["color"], (128,128,128)), 1)
        # cv2.imshow("All Classified Circles (Hough)", debug_frame_all_circles)
        # --- End debugging drawing ---

        identified_robots_on_frame = identify_robot_patterns(all_classified_circles, frame)
        
        cv2.imshow('Robot Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()