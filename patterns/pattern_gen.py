import svgwrite
import os

# --- Configuration ---
# Dimensions in cm
CIRCLE_DIAMETER_CM = 4.0
GAP_CM_WITHIN_SQUARE = 0.25  # Gap between circles in the same 2x2 robot pattern
GAP_CM_BETWEEN_SQUARES = 1.0 # Gap between different 2x2 robot patterns
PADDING_CM = 1.0             # Padding around the entire pattern

# Colors (bright and distinct)
COLORS = {
    "blue": "#007BFF",    # Bright Blue
    "yellow": "#FFEB3B",  # Bright Yellow
    "pink": "#FF4081",    # Bright Pink
}

# Define 4 unique and asymmetrical color patterns for groups of 4 circles
# (TopLeft, TopRight, BottomLeft, BottomRight) for each 2x2 robot pattern
ROBOT_PATTERNS = [
    (COLORS["blue"], COLORS["yellow"], COLORS["pink"], COLORS["blue"]),    # Robot 1 (Top-Left in overall grid)
    (COLORS["yellow"], COLORS["pink"], COLORS["blue"], COLORS["yellow"]),  # Robot 2 (Top-Right in overall grid)
    (COLORS["pink"], COLORS["blue"], COLORS["yellow"], COLORS["pink"]),    # Robot 3 (Bottom-Left in overall grid)
    (COLORS["blue"], COLORS["yellow"], COLORS["blue"], COLORS["pink"])     # Robot 4 (Bottom-Right in overall grid)
]

NUM_ROBOT_PATTERNS_TOTAL = len(ROBOT_PATTERNS) # Should be 4 for a 2x2 grid
PATTERNS_PER_ROW_COL = 2 # For the overall 2x2 grid of patterns

# DPI for cm to px conversion
DPI = 96
CM_TO_INCH = 1 / 2.54

def cm_to_px(cm):
    return cm * CM_TO_INCH * DPI

# --- Calculations ---
circle_diameter_px = cm_to_px(CIRCLE_DIAMETER_CM)
circle_radius_px = circle_diameter_px / 2
gap_within_square_px = cm_to_px(GAP_CM_WITHIN_SQUARE)
gap_between_squares_px = cm_to_px(GAP_CM_BETWEEN_SQUARES)
padding_px = cm_to_px(PADDING_CM)

# Dimensions of one 2x2 robot pattern
single_robot_pattern_width_px = (2 * circle_diameter_px) + gap_within_square_px
single_robot_pattern_height_px = (2 * circle_diameter_px) + gap_within_square_px

# Total width of the SVG (2 patterns wide + 1 gap between them + padding)
total_width_px = (PATTERNS_PER_ROW_COL * single_robot_pattern_width_px) + \
                 ((PATTERNS_PER_ROW_COL - 1) * gap_between_squares_px) + \
                 (2 * padding_px)

# Total height of the SVG (2 patterns high + 1 gap between them + padding)
total_height_px = (PATTERNS_PER_ROW_COL * single_robot_pattern_height_px) + \
                  ((PATTERNS_PER_ROW_COL - 1) * gap_between_squares_px) + \
                  (2 * padding_px)

# --- SVG Generation ---
output_filename = "circle_pattern_overall_2x2.svg"
dwg = svgwrite.Drawing(output_filename, size=(f"{total_width_px}px", f"{total_height_px}px"), profile='tiny')

pattern_index = 0
for row in range(PATTERNS_PER_ROW_COL):
    for col in range(PATTERNS_PER_ROW_COL):
        if pattern_index >= NUM_ROBOT_PATTERNS_TOTAL:
            break # Should not happen if NUM_ROBOT_PATTERNS_TOTAL is 4

        pattern_colors = ROBOT_PATTERNS[pattern_index]

        # Calculate top-left corner X, Y for the current 2x2 robot pattern
        current_robot_pattern_start_x = padding_px + (col * (single_robot_pattern_width_px + gap_between_squares_px))
        current_robot_pattern_start_y = padding_px + (row * (single_robot_pattern_height_px + gap_between_squares_px))

        # Calculate center coordinates for the 4 circles within this robot pattern
        # Relative to current_robot_pattern_start_x and current_robot_pattern_start_y
        # Top-Left circle
        c1_x = current_robot_pattern_start_x + circle_radius_px
        c1_y = current_robot_pattern_start_y + circle_radius_px
        
        # Top-Right circle
        c2_x = current_robot_pattern_start_x + circle_diameter_px + gap_within_square_px + circle_radius_px
        c2_y = current_robot_pattern_start_y + circle_radius_px
        
        # Bottom-Left circle
        c3_x = current_robot_pattern_start_x + circle_radius_px
        c3_y = current_robot_pattern_start_y + circle_diameter_px + gap_within_square_px + circle_radius_px
        
        # Bottom-Right circle
        c4_x = current_robot_pattern_start_x + circle_diameter_px + gap_within_square_px + circle_radius_px
        c4_y = current_robot_pattern_start_y + circle_diameter_px + gap_within_square_px + circle_radius_px

        centers_and_colors = [
            ((c1_x, c1_y), pattern_colors[0]), # Top-Left of individual pattern
            ((c2_x, c2_y), pattern_colors[1]), # Top-Right of individual pattern
            ((c3_x, c3_y), pattern_colors[2]), # Bottom-Left of individual pattern
            ((c4_x, c4_y), pattern_colors[3])  # Bottom-Right of individual pattern
        ]

        for (center_x, center_y), color in centers_and_colors:
            dwg.add(dwg.circle(center=(center_x, center_y), r=circle_radius_px, fill=color))
        
        pattern_index += 1
    if pattern_index >= NUM_ROBOT_PATTERNS_TOTAL:
        break

dwg.save()

print(f"SVG file '{output_filename}' generated successfully in '{os.getcwd()}'.")
print(f"Dimensions: {total_width_px:.2f}px x {total_height_px:.2f}px")
print(f"Equivalent CM: {(total_width_px / DPI / CM_TO_INCH):.2f}cm x {(total_height_px / DPI / CM_TO_INCH):.2f}cm")
