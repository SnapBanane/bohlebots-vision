import cv2
import numpy as np
import math

# --- Configuration ---
DRAW_COLORS = {
    "blue": (255, 123, 0),    # BGR for drawing
    "yellow": (59, 235, 255), # BGR for drawing
    "pink": (129, 64, 255)    # BGR for drawing
}

ROBOT_PATTERNS_DEF = {
    "Robot 1": ("blue", "yellow", "pink", "blue"),
    "Robot 2": ("yellow", "pink", "blue", "yellow"),
    "Robot 3": ("pink", "blue", "yellow", "pink"),
    "Robot 4": ("blue", "yellow", "blue", "pink")
}

# --- SVG Input Configuration ---
SVG_FILE_PATH = "patterns/circle_pattern.svg" # <<<< IMPORTANT: Set this to your SVG file path

# Define the *exact* BGR colors used in your SVG for the colored circles.
# Ensure these BGR values precisely match what's in your SVG file.
SVG_TARGET_COLORS_BGR = {
    "blue": (255, 0, 0),      # Pure Blue (BGR)
    "yellow": (0, 255, 255),  # Pure Yellow (BGR)
    "pink": (255, 0, 255)     # Pure Magenta (BGR) - adjust if your SVG uses a different pink
}

def generate_hsv_ranges_for_svg(bgr_color_dict, h_tolerance=10, s_tolerance=50, v_tolerance=50):
    """
    Generates tight HSV ranges for exact BGR colors.
    Tolerances increased to account for black borders and averaging effects.
    """
    hsv_ranges = {}
    for name, bgr_tuple in bgr_color_dict.items():
        b, g, r = bgr_tuple
        # Create a 1x1 pixel image in BGR
        pixel_bgr = np.uint8([[[b, g, r]]])
        # Convert to HSV
        pixel_hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = pixel_hsv[0][0]

        # Define tight ranges around the exact HSV value
        lower_h = max(0, h - h_tolerance)
        upper_h = min(179, h + h_tolerance) # Max H value in OpenCV is 179
        
        # Corrected calculation to prevent overflow before min()
        lower_s = max(0, int(s) - s_tolerance)
        upper_s = min(255, int(s) + s_tolerance)
        lower_v = max(0, int(v) - v_tolerance)
        upper_v = min(255, int(v) + v_tolerance)
        
        # Handle specific cases like red which can wrap around H=0/179
        # For pure blue, yellow, magenta, direct range is usually fine.
        # If you use a red color (H near 0 or 179), you might need two ranges or special logic.

        hsv_ranges[name] = (np.array([lower_h, lower_s, lower_v]), np.array([upper_h, upper_s, upper_v]))
        print(f"Generated HSV range for SVG color '{name}' (H:{h} S:{s} V:{v}): {hsv_ranges[name][0]} to {hsv_ranges[name][1]}")
    return hsv_ranges

# --- HSV Color Ranges (Generated for SVG colors) ---
# These ranges will be specific to the exact colors in your SVG.
HSV_RANGES = generate_hsv_ranges_for_svg(SVG_TARGET_COLORS_BGR)


# --- Detection Parameters ---
# These might need tuning based on the scale and rendering of your SVG
MIN_CIRCLE_RADIUS_PX = 20  # Reduced minimum radius
MAX_CIRCLE_RADIUS_PX = 60  # Reduced maximum radius to be more specific
HOUGH_PARAM1 = 50  # Edge detection threshold
HOUGH_PARAM2 = 15  # Lowered for more sensitive detection (was 20)
HOUGH_MIN_DIST_BETWEEN_CIRCLES = 50  # Reduced minimum distance between circles

def classify_circle_color(hsv_frame, center, radius, debug=False):
    """
    Calculates the average HSV color within a circular region and classifies it.
    Returns the color name (str) or None if not classified.
    """
    mask = np.zeros(hsv_frame.shape[:2], dtype="uint8")
    cv2.circle(mask, center, int(radius), 255, -1) # Ensure radius is int
    
    # Ensure we don't have an empty mask if circle is off-image or too small
    if np.sum(mask) == 0:
        if debug:
            print(f"    Empty mask for circle at {center} with radius {radius}")
        return None

    mean_hsv_tuple = cv2.mean(hsv_frame, mask=mask)
    # mean_hsv_tuple will have 4 values (H,S,V,Alpha if present), we only need first 3
    h, s, v = int(mean_hsv_tuple[0]), int(mean_hsv_tuple[1]), int(mean_hsv_tuple[2])

    if debug:
        print(f"    Circle at {center}: avg HSV = ({h}, {s}, {v})")

    # Check against defined HSV ranges
    for color_name, (lower, upper) in HSV_RANGES.items():
        if debug:
            print(f"      Checking {color_name}: {lower} <= ({h}, {s}, {v}) <= {upper}")
        if lower[0] <= h <= upper[0] and \
           lower[1] <= s <= upper[1] and \
           lower[2] <= v <= upper[2]:
            if debug:
                print(f"      ✅ Classified as {color_name}")
            return color_name
        elif debug:
            print(f"      ❌ Not {color_name}")
        
    if debug:
        print(f"      ❌ No color match found")
    return None


def find_and_classify_circles(frame, hsv_frame, debug=False):
    """
    Detects circles using Hough Transform, then classifies their color by averaging.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try with slight blur to help with edge detection
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    if debug:
        print(f"\n=== CIRCLE DETECTION DEBUG ===")
        print(f"Image size: {frame.shape}")
        print(f"Detection parameters:")
        print(f"  minRadius={MIN_CIRCLE_RADIUS_PX}, maxRadius={MAX_CIRCLE_RADIUS_PX}")
        print(f"  param1={HOUGH_PARAM1}, param2={HOUGH_PARAM2}")
        print(f"  minDist={HOUGH_MIN_DIST_BETWEEN_CIRCLES}")

    detected_circles_hough = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,  # Accumulator resolution
        minDist=HOUGH_MIN_DIST_BETWEEN_CIRCLES,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=MIN_CIRCLE_RADIUS_PX,
        maxRadius=MAX_CIRCLE_RADIUS_PX    )
    
    classified_circles_list = []
    if detected_circles_hough is not None:
        detected_circles_hough = np.uint16(np.around(detected_circles_hough))
        if debug:
            print(f"Hough detected {len(detected_circles_hough[0])} circles")
        
        for i, pt in enumerate(detected_circles_hough[0, :]):
            center_x, center_y, radius = int(pt[0]), int(pt[1]), int(pt[2])
            
            if debug:
                print(f"  Circle {i+1}: center=({center_x}, {center_y}), radius={radius}")
            
            if radius == 0:
                if debug:
                    print(f"    ❌ Skipping zero radius circle")
                continue

            color_name = classify_circle_color(hsv_frame, (center_x, center_y), radius, debug)
            if color_name:
                classified_circles_list.append({
                    "center": (center_x, center_y),
                    "radius": radius,
                    "color": color_name
                })
                if debug:
                    print(f"    ✅ Added to classified list as {color_name}")
            elif debug:
                print(f"    ❌ Could not classify color")
    else:
        if debug:
            print(f"❌ No circles found with Hough transform")
    
    if debug:
        print(f"Final result: {len(classified_circles_list)} classified circles")
    
    return classified_circles_list

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_midpoint(p1, p2):
    return ( (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 )

def identify_robot_patterns(all_detected_circles, frame_to_draw_on, debug=False):
    """
    Identifies 2x2 robot patterns from a list of all detected circles.
    Simplified and faster approach with extensive debugging.
    """
    if debug:
        print(f"\n=== PATTERN DETECTION DEBUG ===")
        print(f"Input: {len(all_detected_circles)} circles")
        for i, circle in enumerate(all_detected_circles):
            print(f"  Circle {i+1}: {circle['color']} at {circle['center']} radius={circle['radius']}")
    
    if len(all_detected_circles) < 4:
        if debug:
            print(f"❌ Not enough circles for pattern detection: {len(all_detected_circles)}/4")
        return []

    identified_robots = []
    from itertools import combinations
    
    # Expected spacing from our animation
    expected_spacing = 80
    tolerance = 25  # Reduced tolerance for faster processing
    
    potential_groups = combinations(all_detected_circles, 4)
    group_count = 0

    for group_of_4_tuple in potential_groups:
        group_of_4 = list(group_of_4_tuple)
        group_count += 1
        
        if debug:
            print(f"\n--- Testing group {group_count} ---")
            print(f"Circles: {[c['color'] for c in group_of_4]}")
            print(f"Positions: {[c['center'] for c in group_of_4]}")
        
        # Simple approach: check if any 4 circles form a reasonable square/rectangle
        # Calculate center of the 4 circles
        centers = [c["center"] for c in group_of_4]
        center_x = sum(pos[0] for pos in centers) / 4
        center_y = sum(pos[1] for pos in centers) / 4
        
        if debug:
            print(f"Group center: ({center_x:.1f}, {center_y:.1f})")
        
        # Check if all circles are roughly the expected distance from center
        # For a square with side length 80, distance from center to corner is about 80*sqrt(2)/2 ≈ 56
        expected_center_dist = expected_spacing * 0.7  # Approximately 56
        if debug:
            print(f"Expected distance from center: {expected_center_dist:.1f} ± {tolerance}")
        
        valid_square = True
        distances = []
        for i, circle in enumerate(group_of_4):
            dist_from_center = distance(circle["center"], (center_x, center_y))
            distances.append(dist_from_center)
            if debug:
                print(f"  Circle {i+1} ({circle['color']}): distance = {dist_from_center:.1f}")
            if abs(dist_from_center - expected_center_dist) > tolerance:
                if debug:
                    print(f"    ❌ Outside tolerance! |{dist_from_center:.1f} - {expected_center_dist:.1f}| = {abs(dist_from_center - expected_center_dist):.1f} > {tolerance}")
                valid_square = False
            else:
                if debug:
                    print(f"    ✅ Within tolerance")
        
        if not valid_square:
            if debug:
                print(f"❌ Group {group_count}: Invalid square geometry")
            continue
        
        if debug:
            print(f"✅ Group {group_count}: Valid square geometry!")
        
        # If we have a valid square, determine the arrangement by position
        # Sort by angle from center to determine TL, TR, BL, BR
        arranged = [None, None, None, None]  # TL, TR, BL, BR
        
        if debug:
            print("Quadrant assignment:")
        for i, circle in enumerate(group_of_4):
            dx = circle["center"][0] - center_x
            dy = circle["center"][1] - center_y
            
            if debug:
                print(f"  Circle {i+1} ({circle['color']}): dx={dx:.1f}, dy={dy:.1f}", end="")
            
            if dx < 0 and dy < 0:  # Top-left quadrant
                arranged[0] = circle
                if debug: print(" -> TL")
            elif dx > 0 and dy < 0:  # Top-right quadrant
                arranged[1] = circle
                if debug: print(" -> TR")
            elif dx < 0 and dy > 0:  # Bottom-left quadrant
                arranged[2] = circle
                if debug: print(" -> BL")
            elif dx > 0 and dy > 0:  # Bottom-right quadrant
                arranged[3] = circle
                if debug: print(" -> BR")
            else:
                if debug: print(" -> ??? (on axis)")
          # Check if we have all 4 positions filled
        if None in arranged:
            if debug:
                print(f"❌ Group {group_count}: Missing quadrant assignments")
                print(f"Arrangement: TL={arranged[0] is not None}, TR={arranged[1] is not None}, BL={arranged[2] is not None}, BR={arranged[3] is not None}")
            continue
        
        if debug:
            print(f"✅ All quadrants assigned!")
        sorted_group = arranged
        
        current_pattern_colors = (
            sorted_group[0]["color"],  # TL
            sorted_group[1]["color"],  # TR
            sorted_group[2]["color"],  # BL
            sorted_group[3]["color"]   # BR
        )
        
        if debug:
            print(f"Pattern found: TL={current_pattern_colors[0]}, TR={current_pattern_colors[1]}, BL={current_pattern_colors[2]}, BR={current_pattern_colors[3]}")
            print(f"Checking against known patterns (with rotation):")
        
        pattern_match = False
        matched_robot_name = None
        
        # Check all possible rotations of the pattern (0°, 90°, 180°, 270°)
        for robot_name, defined_pattern_colors in ROBOT_PATTERNS_DEF.items():
            if debug:
                print(f"  {robot_name}: {defined_pattern_colors}")
            
            # Generate all 4 rotations of the current pattern
            # Rotation mappings: TL->TR->BR->BL->TL
            rotations = [
                current_pattern_colors,  # 0° (original)
                (current_pattern_colors[2], current_pattern_colors[0], current_pattern_colors[3], current_pattern_colors[1]),  # 90° CW
                (current_pattern_colors[3], current_pattern_colors[2], current_pattern_colors[1], current_pattern_colors[0]),  # 180°
                (current_pattern_colors[1], current_pattern_colors[3], current_pattern_colors[0], current_pattern_colors[2])   # 270° CW
            ]
            
            for rotation_angle, rotated_pattern in enumerate(rotations):
                if debug and rotation_angle == 0:
                    print(f"    Trying rotations: 0°={rotated_pattern}")
                elif debug:
                    print(f"                      {rotation_angle*90}°={rotated_pattern}")
                
                if rotated_pattern == defined_pattern_colors:
                    if debug:
                        print(f"  ✅ MATCH! Found {robot_name} (rotated {rotation_angle*90}°)")
                    pattern_match = True
                    matched_robot_name = robot_name
                    break
            
            if pattern_match:
                break
        
        if pattern_match:
            square_center = (int(center_x), int(center_y))

            # Calculate orientation vector (front is between TL and TR)
            p_tl = sorted_group[0]["center"]
            p_tr = sorted_group[1]["center"]
            front_midpoint = get_midpoint(p_tl, p_tr)
            
            avg_radius_val = sum(c['radius'] for c in sorted_group) / 4.0
            
            # Create bounding box
            all_x_coords = [c["center"][0] for c in sorted_group]
            all_y_coords = [c["center"][1] for c in sorted_group]
            min_x = min(all_x_coords) - int(avg_radius_val)
            max_x = max(all_x_coords) + int(avg_radius_val)
            min_y = min(all_y_coords) - int(avg_radius_val)
            max_y = max(all_y_coords) + int(avg_radius_val)
            bbox = (min_x, min_y, max_x - min_x, max_y - min_y)

            # Check if this robot is too close to an existing one
            is_new_robot = True
            for existing_robot in identified_robots:
                dist_centers = distance(square_center, existing_robot["center"])
                if dist_centers < expected_spacing: 
                    if debug:
                        print(f"  ❌ Too close to existing robot at {existing_robot['center']} (distance: {dist_centers:.1f})")
                    is_new_robot = False
                    break
            
            if not is_new_robot:
                continue

            identified_robots.append({
                "name": matched_robot_name,
                "center": square_center,
                "orientation_vector_start": square_center,
                "orientation_vector_end": front_midpoint,
                "circles_in_pattern": sorted_group, 
                "bbox": bbox
            })
            
            # Drawing
            cv2.rectangle(frame_to_draw_on, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0,255,0), 2)
            cv2.putText(frame_to_draw_on, matched_robot_name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.arrowedLine(frame_to_draw_on, square_center, front_midpoint, (255,0,0), 2, tipLength=0.2)
            
            for circle_data in sorted_group:
                center_pt = (int(circle_data["center"][0]), int(circle_data["center"][1]))
                radius_val = int(circle_data["radius"])
                cv2.circle(frame_to_draw_on, center_pt, radius_val, DRAW_COLORS.get(circle_data["color"], (128,128,128)), -1)
                cv2.circle(frame_to_draw_on, center_pt, radius_val, (0,0,0), 1)
        
        if debug and not pattern_match:
            print(f"  ❌ No pattern match found")
    
    if debug:
        print(f"\n=== FINAL RESULT: {len(identified_robots)} robots detected ===")
    return identified_robots


def create_animated_test_image(frame_count):
    """
    Create an animated test image with multiple robots moving and rotating around the screen.
    """
    # Create a white background image
    img_width, img_height = 1200, 800
    frame = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255  # White background
    
    # Define circle properties
    circle_radius = 30
    spacing = 80  # Distance between circle centers in robot pattern
    
    # Define multiple robots with different patterns and movement
    robots_config = [
        {
            "name": "Robot 1",
            "pattern": ["blue", "yellow", "pink", "blue"],
            "center_path": "circular",
            "path_radius": 200,
            "path_center": (300, 400),
            "speed": 0.02,
            "rotation_speed": 0.01,
            "phase_offset": 0
        },
        {
            "name": "Robot 2", 
            "pattern": ["yellow", "pink", "blue", "yellow"],
            "center_path": "figure_eight",
            "path_radius": 150,
            "path_center": (600, 400),
            "speed": 0.03,
            "rotation_speed": -0.015,
            "phase_offset": math.pi
        },
        {
            "name": "Robot 3",
            "pattern": ["pink", "blue", "yellow", "pink"],
            "center_path": "linear",
            "path_radius": 100,
            "path_center": (900, 400),
            "speed": 0.025,
            "rotation_speed": 0.02,
            "phase_offset": math.pi/2
        },
        {
            "name": "Robot 4",
            "pattern": ["blue", "yellow", "blue", "pink"],
            "center_path": "spiral",
            "path_radius": 120,
            "path_center": (600, 200),
            "speed": 0.01,
            "rotation_speed": 0.03,
            "phase_offset": 3*math.pi/2
        }
    ]
    
    for robot in robots_config:
        # Calculate robot center position based on movement pattern
        t = frame_count * robot["speed"] + robot["phase_offset"]
        
        if robot["center_path"] == "circular":
            robot_center_x = int(robot["path_center"][0] + robot["path_radius"] * math.cos(t))
            robot_center_y = int(robot["path_center"][1] + robot["path_radius"] * math.sin(t))
        elif robot["center_path"] == "figure_eight":
            robot_center_x = int(robot["path_center"][0] + robot["path_radius"] * math.cos(t))
            robot_center_y = int(robot["path_center"][1] + robot["path_radius"] * math.sin(2*t) * 0.5)
        elif robot["center_path"] == "linear":
            robot_center_x = int(robot["path_center"][0] + robot["path_radius"] * math.cos(t))
            robot_center_y = int(robot["path_center"][1] + 50 * math.sin(t*3))  # Small vertical oscillation
        elif robot["center_path"] == "spiral":
            spiral_radius = robot["path_radius"] * (0.5 + 0.5 * math.sin(t*0.3))
            robot_center_x = int(robot["path_center"][0] + spiral_radius * math.cos(t*2))
            robot_center_y = int(robot["path_center"][1] + spiral_radius * math.sin(t*2))
        
        # Calculate robot orientation (rotation)
        rotation_angle = frame_count * robot["rotation_speed"] + robot["phase_offset"]
        
        # Define the 2x2 pattern positions relative to robot center
        # TL, TR, BL, BR with rotation
        relative_positions = [
            (-spacing/2, -spacing/2),  # Top-left
            (spacing/2, -spacing/2),   # Top-right  
            (-spacing/2, spacing/2),   # Bottom-left
            (spacing/2, spacing/2)     # Bottom-right
        ]
        
        # Apply rotation to relative positions
        cos_rot = math.cos(rotation_angle)
        sin_rot = math.sin(rotation_angle)
        
        positions = []
        for rel_x, rel_y in relative_positions:
            # Rotate the relative position
            rotated_x = rel_x * cos_rot - rel_y * sin_rot
            rotated_y = rel_x * sin_rot + rel_y * cos_rot
            
            # Add to robot center
            abs_x = int(robot_center_x + rotated_x)
            abs_y = int(robot_center_y + rotated_y)
            positions.append((abs_x, abs_y))
        
        # Draw circles with pattern colors
        for i, (pos, color_name) in enumerate(zip(positions, robot["pattern"])):
            # Check if position is within image bounds
            if 0 <= pos[0] < img_width and 0 <= pos[1] < img_height:
                bgr_color = SVG_TARGET_COLORS_BGR[color_name]
                cv2.circle(frame, pos, circle_radius, bgr_color, -1)  # Filled circle
    
    return frame

def show_final_mask(hsv_frame):
    """
    Create and display a final combined mask showing all detected colors.
    Returns a dictionary of masks for each color.
    """
    masks = {}
    
    # Create masks for each color
    for color_name, (lower, upper) in HSV_RANGES.items():
        mask = cv2.inRange(hsv_frame, lower, upper)
        masks[color_name] = mask
    
    # Create a combined mask showing all colors
    combined_mask = np.zeros_like(hsv_frame)
    for color_name, mask in masks.items():
        # Use the drawing colors for the combined view
        bgr_color = DRAW_COLORS.get(color_name, (128, 128, 128))
        combined_mask[mask > 0] = bgr_color
    
    cv2.imshow("Final Combined HSV Mask", combined_mask)
    return masks

def show_hsv_masks(hsv_frame):
    """
    Create and display individual HSV masks for each color for debugging.
    """
    masks = {}
    for color_name, (lower, upper) in HSV_RANGES.items():
        mask = cv2.inRange(hsv_frame, lower, upper)
        masks[color_name] = mask
        
        # Show individual mask
        cv2.imshow(f"HSV Mask - {color_name}", mask)
    
    # Create a combined mask showing all colors
    combined_mask = np.zeros_like(hsv_frame)
    for color_name, mask in masks.items():
        # Use the drawing colors for the combined view
        bgr_color = DRAW_COLORS.get(color_name, (128, 128, 128))
        combined_mask[mask > 0] = bgr_color
    
    cv2.imshow("Combined HSV Masks", combined_mask)
    return masks

def main():
    # Create animated test with multiple robots
    frame_count = 0
    print("Starting animated robot detection test...")
    print("Press 'q' to quit, 'p' to pause/unpause, SPACE to step frame when paused")
    print("Press 'd' to toggle detailed debugging")
    
    paused = False
    debug_enabled = False  # Toggle for detailed debugging
    
    try:
        while True:
            # Create the current frame
            frame = create_animated_test_image(frame_count)
            
            if frame is None:
                print("Error: Could not create test image.")
                return
              # Create a copy for drawing, so the original frame isn't modified
            display_frame = frame.copy()

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
              # Enable detailed debugging based on user toggle or periodically
            debug_patterns = debug_enabled or (frame_count % 120 == 0)  # Every 4 seconds when enabled
            
            # Show final combined mask for debugging (optional - commented out for performance)
            # masks = show_final_mask(hsv_frame)
            
            all_classified_circles = find_and_classify_circles(frame, hsv_frame, debug_patterns)
            
            # Show HSV masks for debugging when detailed debugging is enabled
            if debug_patterns and debug_enabled:
                show_hsv_masks(hsv_frame)
            
            identified_robots_on_frame = identify_robot_patterns(all_classified_circles, display_frame, debug_patterns)
              # Print robot detection results with vectors (less frequent to avoid spam)
            if frame_count % 60 == 0:  # Print every 60 frames (every 2 seconds)
                print(f"\n=== FRAME {frame_count} SUMMARY ===")
                print(f"Total circles detected: {len(all_classified_circles)}")
                print(f"Robots identified: {len(identified_robots_on_frame)}")
                
                if len(identified_robots_on_frame) > 0:
                    print("Robot details:")
                    for i, robot in enumerate(identified_robots_on_frame):
                        center = robot['center']
                        orientation_end = robot['orientation_vector_end']
                        # Calculate vector direction
                        dx = orientation_end[0] - center[0]
                        dy = orientation_end[1] - center[1]
                        angle_rad = math.atan2(dy, dx)
                        angle_deg = math.degrees(angle_rad)
                        print(f"  {robot['name']}: center={center}, orientation={angle_deg:.1f}°")
                else:
                    print("  No robots detected in this frame")
                print("=" * 50)
            
            # Add frame counter to the display
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(display_frame, f"Robots: {len(identified_robots_on_frame)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            if paused:
                cv2.putText(display_frame, "PAUSED - Press 'p' to continue, SPACE for next frame", 
                           (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Live Robot Detection Test', display_frame)            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF  # 30ms delay for smooth animation
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"Animation {'paused' if paused else 'resumed'}")
            elif key == ord('d'):
                debug_enabled = not debug_enabled
                print(f"Detailed debugging {'enabled' if debug_enabled else 'disabled'}")
            elif key == ord(' ') and paused:
                # Step one frame when paused
                frame_count += 1
                continue
                
            # Increment frame counter only if not paused
            if not paused:
                frame_count += 1

        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error in main: {e}")
        return

if __name__ == '__main__':
    main()