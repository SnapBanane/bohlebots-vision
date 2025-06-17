import cv2 as cv
import numpy as np
import imutils
import math
import depthai as dai


color = (255,255,255)
# Updated color boundaries with your specific values
colors = {'blue': [np.array([84, 161, 45]), np.array([124, 255, 205])],    # Your sampled blue circle
          'pink': [np.array([148, 38, 57]), np.array([179, 198, 217])],    # From your existing ranges
          'yellow': [np.array([5, 21, 65]), np.array([45, 181, 225])],     # From your existing ranges
          'red': [np.array([0, 120, 70]), np.array([10, 255, 255])],       # Standard red range
          'green': [np.array([40, 80, 50]), np.array([80, 255, 255])]      # Standard green range
          }

# Robot pattern definitions
robot_patterns = {
    'robot1': {
        'front_colors': ['blue', 'blue'], 
        'back_left_color': 'yellow',     
        'back_right_color': 'pink',      
        'min_circles': 3,                
        'id': 1
    }
}


def create_oak_pipeline():
    """Create and configure DepthAI pipeline for OAK camera"""
    # Create pipeline
    pipeline = dai.Pipeline()

    # Define source and output
    camRgb = pipeline.create(dai.node.ColorCamera)
    xout = pipeline.create(dai.node.XLinkOut)

    xout.setStreamName("rgb")

    # Properties
    camRgb.setPreviewSize(1920, 1080)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)  # Changed to BGR

    # Linking
    camRgb.preview.link(xout.input)

    return pipeline


def is_circle_like(contour):
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    if perimeter == 0:
        return False
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    if 0.7 <= circularity <= 1.2:
        return True
    return False


def filter_similar_sizes(detected_objects, tolerance=0.5):
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
        if 100 <= area <= 20000 and is_circle_like(c):
            M = cv.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                detected_objects.append((c, cx, cy))
    
    filtered_objects = filter_similar_sizes(detected_objects)
    return filtered_objects


def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_robot_pose_simple(blue_circles, yellow_circle, pink_circle):
    """
    Simple robust calculation of robot pose.
    Priority order for heading calculation:
    1. If 2 blues + yellow + pink: heading from back-center to front-center
    2. If 2 blues + 1 back: heading from single back to front-center  
    3. If 1 blue + yellow + pink: heading from back-center to single blue
    """
    
    # Collect all available circles
    all_circles = []
    if blue_circles:
        all_circles.extend(blue_circles)
    if yellow_circle:
        all_circles.append(yellow_circle)
    if pink_circle:
        all_circles.append(pink_circle)
    
    if len(all_circles) < 3:
        return None, None, 0
    
    # Robot position is always the centroid of all detected circles
    robot_x = sum(circle[0] for circle in all_circles) / len(all_circles)
    robot_y = sum(circle[1] for circle in all_circles) / len(all_circles)
    robot_pos = (robot_x, robot_y)
    
    # Calculate heading based on available circles
    heading_angle = None
    
    # Case 1: Two blues available (ideal case)
    if len(blue_circles) == 2:
        front_center_x = (blue_circles[0][0] + blue_circles[1][0]) / 2
        front_center_y = (blue_circles[0][1] + blue_circles[1][1]) / 2
        
        # If we have both back circles
        if yellow_circle and pink_circle:
            back_center_x = (yellow_circle[0] + pink_circle[0]) / 2
            back_center_y = (yellow_circle[1] + pink_circle[1]) / 2
            heading_angle = math.atan2(front_center_y - back_center_y, front_center_x - back_center_x)
        
        # If we have only one back circle
        elif yellow_circle:
            heading_angle = math.atan2(front_center_y - yellow_circle[1], front_center_x - yellow_circle[0])
        elif pink_circle:
            heading_angle = math.atan2(front_center_y - pink_circle[1], front_center_x - pink_circle[0])
    
    # Case 2: One blue available
    elif len(blue_circles) == 1 and yellow_circle and pink_circle:
        blue_pos = blue_circles[0]
        back_center_x = (yellow_circle[0] + pink_circle[0]) / 2
        back_center_y = (yellow_circle[1] + pink_circle[1]) / 2
        heading_angle = math.atan2(blue_pos[1] - back_center_y, blue_pos[0] - back_center_x)
    
    return robot_pos, heading_angle, len(all_circles)


def find_robot_clusters(blue_positions, yellow_positions, pink_positions, max_cluster_distance=150):
    """
    Find clusters of circles that could form a robot.
    Returns the best cluster based on geometric constraints.
    """
    best_cluster = None
    best_score = 0
    
    # Try all combinations of available circles
    for num_blues in range(min(3, len(blue_positions) + 1)):
        for num_yellows in range(min(2, len(yellow_positions) + 1)):
            for num_pinks in range(min(2, len(pink_positions) + 1)):
                
                # Skip if we don't have enough circles
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
                            
                            # Collect all circles in this combination
                            cluster_circles = list(blue_combo) + list(yellow_combo) + list(pink_combo)
                            
                            if len(cluster_circles) < 3:
                                continue
                            
                            # Check if all circles are within reasonable distance of each other
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
    """Simple and robust robot identification"""
    
    blue_positions = all_detected_circles_map.get('blue', [])
    yellow_positions = all_detected_circles_map.get('yellow', [])
    pink_positions = all_detected_circles_map.get('pink', [])
    
    # Find the best cluster of circles
    cluster = find_robot_clusters(blue_positions, yellow_positions, pink_positions)
    
    if not cluster:
        return None
    
    # Extract circles from cluster
    blue_circles = cluster['blues']
    yellow_circle = cluster['yellows'][0] if cluster['yellows'] else None
    pink_circle = cluster['pinks'][0] if cluster['pinks'] else None
    
    # Calculate robot pose
    robot_pos, heading_angle, circle_count = calculate_robot_pose_simple(
        blue_circles, yellow_circle, pink_circle
    )
    
    if robot_pos is None or heading_angle is None:
        return None
    
    # Create back circles list without None values
    back_circles_roles_pos = []
    if yellow_circle:
        back_circles_roles_pos.append(('yellow', yellow_circle))
    if pink_circle:
        back_circles_roles_pos.append(('pink', pink_circle))
    
    return {
        'id': pattern_config['id'],
        'position': robot_pos,
        'heading': heading_angle,
        'front_circles_pos': blue_circles,
        'back_circles_roles_pos': back_circles_roles_pos,
        'circles_count': circle_count,
        'all_circles_pos': blue_circles + ([yellow_circle] if yellow_circle else []) + ([pink_circle] if pink_circle else [])
    }


def main():
    # Create pipeline
    pipeline = create_oak_pipeline()

    # Connect to device and start pipeline
    with dai.Device(pipeline) as device:
        print("OAK camera connected successfully")
        
        # Output queue will be used to get the rgb frames from the output defined above
        q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            inRgb = q.get()  # Blocking call, will wait until a new data has arrived
            
            # Get frame directly in BGR format
            frame = inRgb.getCvFrame()
            
            if frame is None:
                print("Error: Could not read frame from OAK camera.")
                continue
                
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            
            # Detect all circles by color
            all_detected_circles_map = {}
            for name, clr_range in colors.items():
                detected_objects = find_color(hsv, clr_range)
                positions = [(cx, cy) for c, cx, cy in detected_objects]
                all_detected_circles_map[name] = positions
                
                # Draw all detected circles
                for c, cx, cy in detected_objects:
                    cv.drawContours(frame, [c], -1, color, 2)
                    cv.circle(frame, (cx, cy), 5, color, -1)
                    cv.putText(frame, name, (cx-20, cy-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Identify robots
            identified_robots_list = []
            for robot_name_key, pattern in robot_patterns.items():
                robot = identify_robot_from_pattern(all_detected_circles_map, pattern)
                if robot:
                    identified_robots_list.append(robot)
            
            # Draw identified robots
            for robot in identified_robots_list:
                robot_pos = robot['position']
                heading_angle = robot['heading']
                
                # Draw robot center (larger green circle)
                cv.circle(frame, (int(robot_pos[0]), int(robot_pos[1])), 15, (0, 255, 0), -1)
                
                # Draw heading arrow (consistent length)
                arrow_length = 100
                end_x = int(robot_pos[0] + math.cos(heading_angle) * arrow_length)
                end_y = int(robot_pos[1] + math.sin(heading_angle) * arrow_length)
                cv.arrowedLine(frame, (int(robot_pos[0]), int(robot_pos[1])), 
                              (end_x, end_y), (0, 255, 0), 4, tipLength=0.3)
                
                # Draw robot info
                cv.putText(frame, f"Robot {robot['id']}", 
                          (int(robot_pos[0])-40, int(robot_pos[1])-25), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv.putText(frame, f"Circles: {robot['circles_count']}/4", 
                          (int(robot_pos[0])-40, int(robot_pos[1])+40), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv.putText(frame, f"Heading: {math.degrees(heading_angle):.1f}°", 
                          (int(robot_pos[0])-40, int(robot_pos[1])+60), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Highlight robot circles with thicker borders
                for circle_pos in robot['front_circles_pos']:
                    cv.circle(frame, circle_pos, 10, (0, 0, 255), 3)  # Red for front blues
                
                # Fixed: properly handle back circles without None values
                for role, circle_pos in robot['back_circles_roles_pos']:
                    if role == 'yellow':
                        cv.circle(frame, circle_pos, 10, (0, 255, 255), 3)  # Yellow
                    elif role == 'pink':
                        cv.circle(frame, circle_pos, 10, (255, 0, 255), 3)  # Magenta for pink
            
            # Display summary
            cv.putText(frame, f"Robots detected: {len(identified_robots_list)}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv.imshow("OAK Robot Tracking", frame)
            
            if cv.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv.destroyAllWindows()