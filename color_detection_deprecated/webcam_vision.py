import cv2 as cv
import numpy as np
import imutils
import math


color = (255,255,255)
colors = {'blue': [np.array([84, 161, 45]), np.array([124, 255, 205])],
          'pink': [np.array([148, 38, 57]), np.array([179, 198, 217])],
          'yellow': [np.array([5, 21, 65]), np.array([45, 181, 225])],
          'red': [np.array([0, 120, 70]), np.array([10, 255, 255])],
          'green': [np.array([40, 80, 50]), np.array([80, 255, 255])] 
          }

robot_patterns = {
    'robot1': {
        'front_colors': ['blue', 'blue'], 
        'back_left_color': 'yellow',     
        'back_right_color': 'pink',      
        'min_circles': 3,                
        'id': 1
    }
}

def is_circle(contour):
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return False
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if 0.7 <= circularity <= 1.2:
        return True
    return False


def filter_similar_sizes(detected_objects, tolerance=0.7):
    if len(detected_objects) < 2:
        return detected_objects
    
    areas = [cv.contourArea(obj[0]) for obj in detected_objects]
    median_area = np.median(areas)
    
    filtered_objects = []
    for i, (c, cx, cy) in enumerate(detected_objects):
        area_ratio = areas[i] / median_area
        if (1 - tolerance) <= area_ratio <= (1 + tolerance):
            filtered_objects.append((c, cx, cy))
    
    return filtered_objects


def find_color(frame, points):
    mask = cv.inRange(frame, points[0], points[1])
    cnts = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    detected_objects = []
    for c in cnts:
        area = cv.contourArea(c)
        if 100 <= area <= 20000 and is_circle(c): # for our field may need to decrease minimum area
            M = cv.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                detected_objects.append((c, cx, cy))
    
    filtered_objects = filter_similar_sizes(detected_objects)
    return filtered_objects


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_robot_pose_simple(blue_circles, yellow_circles, pink_circles):
    all_detected_circles = blue_circles + yellow_circles + pink_circles
    if len(all_detected_circles) < 3: # We need atleast 3 circles to get one certain position
        return None, None, 0

    front_points = [np.array(c['center']) for c in blue_circles]
    back_left_point = np.array(yellow_circles[0]['center']) if yellow_circles else None
    back_right_point = np.array(pink_circles[0]['center']) if pink_circles else None

    points_for_centroid = []
    if front_points: points_for_centroid.extend(front_points)
    if back_left_point is not None: points_for_centroid.append(back_left_point)
    if back_right_point is not None: points_for_centroid.append(back_right_point)

    # If exactly 3 points are visible, estimate the 4th to stabilize the centroid
    if len(points_for_centroid) == 3:
        # Case A: Missing back-right point
        if len(front_points) == 2 and back_left_point is not None:
            f1, f2 = front_points
            # Assign left/right front based on proximity to the known back-left point
            if np.linalg.norm(f1 - back_left_point) < np.linalg.norm(f2 - back_left_point):
                f_left, f_right = f1, f2
            else:
                f_left, f_right = f2, f1
            # Estimate br assuming a parallelogram (vector f_left->bl == f_right->br_est)
            estimated_br = f_right + (back_left_point - f_left)
            points_for_centroid.append(estimated_br)

        # Case B: Missing back-left point
        elif len(front_points) == 2 and back_right_point is not None:
            f1, f2 = front_points
            # Assign left/right front based on proximity to the known back-right point
            if np.linalg.norm(f1 - back_right_point) > np.linalg.norm(f2 - back_right_point):
                f_left, f_right = f1, f2
            else:
                f_left, f_right = f2, f1
            estimated_bl = f_left + (back_right_point - f_right)
            points_for_centroid.append(estimated_bl)

        # Case C: Missing one front point (Definitive Fix using Dot Product)
        elif len(front_points) == 1 and back_left_point is not None and back_right_point is not None:
            f_known = front_points[0]
            
            # Determine if f_known is the left or right front circle using a dot product.
            back_center = (back_left_point + back_right_point) / 2.0
            back_axis_vec = back_right_point - back_left_point
            center_to_f_vec = f_known - back_center
            
            # The sign of the dot product tells us if f_known is on the "left" or "right"
            # side of the normal passing through the back_center.
            dot_product = np.dot(center_to_f_vec, back_axis_vec)
            
            if dot_product < 0:
                # The projection of f_known is on the back_left_point side of the center.
                # Therefore, f_known is the front-left point. Estimate the front-right.
                estimated_f_partner = back_right_point + (f_known - back_left_point)
            else:
                # The projection is on the back_right_point side.
                # Therefore, f_known is the front-right point. Estimate the front-left.
                estimated_f_partner = back_left_point + (f_known - back_right_point)
            
            points_for_centroid.append(estimated_f_partner)

    # The position is the centroid of the (reconstructed) four points.
    robot_pos = tuple(np.mean(points_for_centroid, axis=0)) if points_for_centroid else (None, None)

    heading_vector_raw = None
    if len(front_points) == 2:
        front_center = np.mean(front_points, axis=0)
        front_axis_vec = front_points[1] - front_points[0]
        heading_candidate = np.array([-front_axis_vec[1], front_axis_vec[0]])
        if back_left_point is not None or back_right_point is not None:
            observed_back_center = np.mean([p for p in [back_left_point, back_right_point] if p is not None], axis=0)
            front_to_back_vec = observed_back_center - front_center
            if np.dot(heading_candidate, front_to_back_vec) < 0:
                heading_vector_raw = heading_candidate
            else:
                heading_vector_raw = -heading_candidate
    elif back_left_point is not None and back_right_point is not None:
        back_center = (back_left_point + back_right_point) / 2.0
        back_axis_vec = back_right_point - back_left_point
        heading_candidate = np.array([-back_axis_vec[1], back_axis_vec[0]])
        if len(front_points) > 0:
            observed_front_center = np.mean(front_points, axis=0)
            back_to_front_vec = observed_front_center - back_center
            if np.dot(heading_candidate, back_to_front_vec) > 0:
                heading_vector_raw = heading_candidate
            else:
                heading_vector_raw = -heading_candidate
    elif len(front_points) == 1 and (back_left_point is not None or back_right_point is not None):
        front_center = front_points[0]
        back_center = np.mean([p for p in [back_left_point, back_right_point] if p is not None], axis=0)
        heading_vector_raw = front_center - back_center

    # Calculate angle in radians
    heading_angle = None
    if heading_vector_raw is not None and np.linalg.norm(heading_vector_raw) > 1e-6:
        heading_angle = np.arctan2(heading_vector_raw[1], heading_vector_raw[0])

    return robot_pos, heading_angle, len(all_detected_circles)


def find_robot_clusters(blue_positions, yellow_positions, pink_positions, max_cluster_distance=150):
    best_cluster = None
    best_score = 0
    
    for num_blues in range(min(3, len(blue_positions) + 1)):
        for num_yellows in range(min(2, len(yellow_positions) + 1)):
            for num_pinks in range(min(2, len(pink_positions) + 1)):
                
                if num_blues + num_yellows + num_pinks < 3:
                    continue
                
                # Generate all combinations for this configuration
                from itertools import combinations
                
                blue_combos = list(combinations(blue_positions, num_blues)) if num_blues > 0 else [()]
                yellow_combos = list(combinations(yellow_positions, num_yellows)) if num_yellows > 0 else [()]
                pink_combos = list(combinations(pink_positions, num_pinks)) if num_pinks > 0 else [()]
                
                for blue_combo in blue_combos:
                    for yellow_combo in yellow_combos:
                        for pink_combo in pink_combos:
                            
                            cluster_circles = list(blue_combo) + list(yellow_combo) + list(pink_combo)
                            
                            if len(cluster_circles) < 3:
                                continue
                            
                            #  Calculate distance between all pairs of circles
                            distances = []
                            for i in range(len(cluster_circles)):
                                for j in range(i + 1, len(cluster_circles)):
                                    distances.append(calculate_distance(cluster_circles[i], cluster_circles[j]))
                            
                            max_distance = max(distances) if distances else 0
                            if max_distance > max_cluster_distance:
                                continue
                            
                            # Calculate score (prefer more circles and tighter clusters)
                            score = len(cluster_circles) * 100 - max_distance
                            
                            if score > best_score:
                                best_score = score
                                best_cluster = {
                                    'blues': list(blue_combo),
                                    'yellows': list(yellow_combo),
                                    'pinks': list(pink_combo),
                                    'score': score
                                }
    
    return best_cluster


def identify_robot_from_pattern(all_detected_circles_map, pattern_config):
    front_color = pattern_config['front_colors'][0]
    back_left_color = pattern_config['back_left_color']
    back_right_color = pattern_config['back_right_color']

    blue_circles_raw = all_detected_circles_map.get(front_color, [])
    yellow_circles_raw = all_detected_circles_map.get(back_left_color, [])
    pink_circles_raw = all_detected_circles_map.get(back_right_color, [])

    if isinstance(blue_circles_raw, tuple): blue_circles_raw = blue_circles_raw[0]
    if isinstance(yellow_circles_raw, tuple): yellow_circles_raw = yellow_circles_raw[0]
    if isinstance(pink_circles_raw, tuple): pink_circles_raw = pink_circles_raw[0]

    def standardize_circles(circle_list, color_name):
        standard_list = []
        if not circle_list:
            return standard_list
        for c in circle_list:
            center_point = (c[1], c[2])
            standard_list.append({'center': center_point, 'color': color_name})
        return standard_list

    blue_circles_list = standardize_circles(blue_circles_raw, front_color)
    yellow_circles_list = standardize_circles(yellow_circles_raw, back_left_color)
    pink_circles_list = standardize_circles(pink_circles_raw, back_right_color)
    
    robot_pos, heading_angle, circle_count = calculate_robot_pose_simple(
        blue_circles_list,
        yellow_circles_list,
        pink_circles_list
    )

    if robot_pos is None or circle_count < pattern_config.get('min_circles', 3):
        return None

    return {
        'id': pattern_config['id'],
        'position': robot_pos,
        'angle_rad': heading_angle,
        'circles': circle_count
    }


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        all_detected_circles_map = {}
        for color_name, hsv_range in colors.items():
            circles_found = find_color(hsv_frame, hsv_range)
            all_detected_circles_map[color_name] = circles_found
            for c, cx, cy in circles_found:
                cv.drawContours(frame, [c], -1, (0, 255, 0), 2)

        for name, pattern in robot_patterns.items():
            robot = identify_robot_from_pattern(all_detected_circles_map, pattern)

            if robot:
                heading_angle_rad = robot['angle_rad']
                pos = robot['position']
                
                cv.circle(frame, (int(pos[0]), int(pos[1])), 10, (0, 255, 255), -1)
                cv.putText(frame, f"ID: {robot['id']}", (int(pos[0]) + 15, int(pos[1]) + 5), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if heading_angle_rad is not None:
                    end_x = int(pos[0] + 60 * np.cos(heading_angle_rad))
                    end_y = int(pos[1] + 60 * np.sin(heading_angle_rad))
                    
                    cv.line(frame, (int(pos[0]), int(pos[1])), (end_x, end_y), (0, 255, 0), 3)
                    
                    heading_angle_deg = np.degrees(heading_angle_rad)
                    cv.putText(frame, f"Angle: {heading_angle_deg:.1f}", (int(pos[0]), int(pos[1]) - 20), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv.imshow('Webcam Vision', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()