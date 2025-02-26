import os
import numpy as np
import cv2
from functools import reduce

class TSR:
    def __init__(self, input_videos, output_dir, output_frame_dir, color_ranges, save_times, tolerance=400):
        self.input_videos = input_videos
        self.output_dir = output_dir
        self.output_frame_dir = output_frame_dir
        self.color_ranges = color_ranges
        self.save_times = save_times
        self.tolerance = tolerance
        self.saved_frames = 0
        self.captured_times = set()

    def process_videos(self):
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_frame_dir, exist_ok=True)

        for video_name in self.input_videos:
            input_path = video_name
            video_basename = os.path.splitext(os.path.basename(video_name))[0]
            output_path = os.path.join(self.output_dir, f"52100174_52100369_{video_basename}.mp4")
            output_frame_path = os.path.join(self.output_frame_dir, f"52100174_52100369_{video_basename}")

            # Initialize per-video variables
            self.captured_times = set()
            self.saved_frames = 0
            current_save_times = self.save_times.get(video_name, [])

            input_video = cv2.VideoCapture(input_path)
            if not input_video.isOpened():
                print(f"Cannot open video file: {input_path}")
                continue

            frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(input_video.get(cv2.CAP_PROP_FPS))
            ms_per_frame = 1000 / fps

            output_video = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (frame_width, frame_height)
            )

            while input_video.isOpened():
                ret, frame = input_video.read()
                if not ret:
                    break
                current_time_ms = int(input_video.get(cv2.CAP_PROP_POS_MSEC)) # Get time in milliseconds
                
                # Process frame and find closest save time
                processed_frame = self.process_frame(frame, current_time_ms, output_frame_path, video_basename, current_save_times, ms_per_frame)
                output_video.write(processed_frame)

            input_video.release()
            output_video.release()
            print(f"Finished processing {video_name}")

    def process_frame(self, frame, current_time_ms, output_frame_path, video_name, current_save_times, ms_per_frame):
        height, width, _ = frame.shape
        half_frame = frame[:height // 2, :]

        # Add text
        text = "52100174 52100369"
        cv2.putText(frame, text, (width - 10 - cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0], height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        mask = self.apply_color_filters(half_frame, video_name)
        contours = self.filter_contours(mask)

        self.draw_contours(half_frame, contours)
        self.save_frame_at_times(frame, current_time_ms, output_frame_path, current_save_times, ms_per_frame)
        frame[:height // 2, :] = half_frame

        return frame

    def save_frame_at_times(self, frame, current_time_ms, output_frame_path, current_save_times, ms_per_frame):
        for save_time_ms in current_save_times:
            closest_frame_time = round(current_time_ms / ms_per_frame) * ms_per_frame
            if abs(closest_frame_time - save_time_ms) <= self.tolerance and save_time_ms not in self.captured_times:
                frame_filename = f"{output_frame_path}_{self.saved_frames + 1}.jpg"
                cv2.imwrite(frame_filename, frame)
                self.saved_frames += 1
                self.captured_times.add(save_time_ms)
                break
            
    def apply_color_filters(self, half_frame, video_name):
        blurred_frame = cv2.GaussianBlur(half_frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        
        mask = np.zeros_like(hsv[:, :, 0])
        for color, ranges in self.color_ranges.items():
            if color == 'red':
                mask |= cv2.inRange(hsv, ranges[0], ranges[1]) | cv2.inRange(hsv, ranges[2], ranges[3])
            else:
                mask |= cv2.inRange(hsv, ranges[0], ranges[1])

        # Choose morphological kernel sizes based on video name
        if "video1" in video_name.lower():
            close_kernel = np.ones((5, 5), np.uint8)
            open_kernel = np.ones((3, 3), np.uint8)
        else:
            close_kernel = np.ones((3, 2), np.uint8)
            open_kernel = np.ones((2, 3), np.uint8)
            
        # Apply morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        
        return mask

    def filter_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if not any(
                x_o <= x <= x + w <= x_o + w_o and y_o <= y <= y + h <= y_o + h_o
                for other in contours if cnt is not other for x_o, y_o, w_o, h_o in [cv2.boundingRect(other)]
            ):
                filtered_contours.append(cnt)
        return filtered_contours

    def draw_contours(self, half_frame, contours):
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                x, y, w, h = cv2.boundingRect(cnt)
                roi = half_frame[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                    
                aspect_ratio = w / float(h)
                
                # Draw rectangle for all signs (including triangles and rectangle)
                # Check triangle signs
                if self.is_triangle(cnt) and self.contains_triangle_sign_colors(roi):     
                    # Check the content of signs            
                    if self.detect_sign(roi, "slow_down"):
                        self.add_content_of_sign(half_frame, x, y,w,h, "Slow Down")
                    elif self.detect_sign(roi, "children_crossing"):
                        self.add_content_of_sign(half_frame, x, y,w,h, "Children Crossing")
                
                # Check rectangle signs
                elif 21000 < area < 42000 and self.detect_guide_sign(roi,"guide_sign") :
                    self.add_content_of_sign(half_frame, x, y,w,h, "Guide Sign")

                # Check circle signs
                elif 500 < area < 300000 and 0.8 <= aspect_ratio <= 1.2 and 0.7 < circularity <= 1.2:                  
                    # Check the content of signs
                    if self.detect_do_not_enter(roi):
                        self.add_content_of_sign(half_frame, x, y,w,h, "Do Not Enter")
                    elif self.detect_sign(roi, "keep_right"):
                        self.add_content_of_sign(half_frame, x, y,w,h, "Compulsory Keep Right")
                    elif self.detect_sign(roi, "no_left_turn"):
                        self.add_content_of_sign(half_frame, x, y,w,h, "No Left Turn")
                    elif self.detect_sign(roi, "no_stopping"):
                        self.add_content_of_sign(half_frame, x, y,w,h, "No Stopping")
                    elif self.detect_sign(roi, "no_parking"):
                        self.add_content_of_sign(half_frame, x, y,w,h, "No Parking")

    def detect_guide_sign(self, roi, sign_type):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        height, width = roi.shape[:2]
        total_pixels = height * width

        blue_ranges = [(np.array([95, 150, 0]), np.array([120, 255, 255]))]
        white_ranges = [(np.array([0, 0, 140]), np.array([180, 60, 255]))]

        blue_mask = self.detect_color_regions(hsv, blue_ranges)
        white_mask = self.detect_color_regions(hsv, white_ranges)

        combined_mask = cv2.bitwise_or(blue_mask, white_mask)
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            blue_percentage = self.calculate_percentage(blue_mask, total_pixels)
            white_percentage = self.calculate_percentage(white_mask, total_pixels)

            if sign_type == "guide_sign" and blue_percentage > 0.3 and white_percentage > 0.03:
                print("Detected Guide Sign!")
                return True

        return False

    def detect_color_regions(self, hsv, color_ranges):
        """Helper function to create a mask for a specific color range."""
        if len(color_ranges) == 2:
            mask1 = cv2.inRange(hsv, color_ranges[0][0], color_ranges[0][1])
            mask2 = cv2.inRange(hsv, color_ranges[1][0], color_ranges[1][1])
            return cv2.bitwise_or(mask1, mask2)
        elif len(color_ranges) == 1:
            return cv2.inRange(hsv, color_ranges[0][0], color_ranges[0][1])
        else:
            raise ValueError("color_ranges must contain one or two HSV ranges.")


    def calculate_percentage(self, mask, total_pixels):
        """Calculate the percentage of non-zero pixels in a mask."""
        return np.sum(mask > 0) / total_pixels



    def detect_do_not_enter(self, roi):
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        height, width = roi.shape[:2]
        
        # Define color ranges
        color_ranges = {
            "red": [(np.array([0, 90, 50]), np.array([10, 255, 255])),
                    (np.array([160, 90, 50]), np.array([180, 255, 255]))],
            "blue": [(np.array([100, 150, 0]), np.array([140, 255, 255]))],
            "white": [(np.array([0, 0, 0]), np.array([180, 30, 255]))],
        }
        
        # Generate masks and calculate percentages
        masks = {key: cv2.inRange(hsv, *ranges[0]) for key, ranges in color_ranges.items()}
        masks["red"] = cv2.bitwise_or(cv2.inRange(hsv, *color_ranges["red"][0]), 
                                    cv2.inRange(hsv, *color_ranges["red"][1]))
        percentages = {key: np.sum(mask > 0) / (height * width) for key, mask in masks.items()}
        
        # Process middle section of white mask
        middle_section = masks["white"][height // 3: 2 * height // 3, :]
        kernel = np.ones((3, 3), np.uint8)
        middle_section = cv2.morphologyEx(middle_section, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(middle_section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_white = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_white)
            white_aspect_ratio = w / h if h > 0 else 0
            white_area_percentage = cv2.contourArea(largest_white) / (middle_section.size)
            
            # Check conditions
            if (percentages["red"] > 0.25 and 
                white_aspect_ratio > 2.0 and 
                0.15 < white_area_percentage < 0.5):
                print("Detected Do Not Enter sign!")
                return True
        return False

    def detect_sign(self, roi, sign_type):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        height, width = roi.shape[:2]
        total_pixels = height * width

        # Define color ranges
        color_ranges = {
            "red": [(np.array([0, 90, 50]), np.array([10, 255, 255])),
                    (np.array([160, 90, 50]), np.array([180, 255, 255]))],
            "blue": [(np.array([90, 50, 50]), np.array([160, 255, 255]))],
            "white": [(np.array([0, 0, 100]), np.array([180, 60, 220]))],
            "black": [(np.array([0, 0, 0]), np.array([180, 150, 100]))],
            "yellow": [(np.array([10, 70, 70]), np.array([35, 255, 255]))],
            "yellow_2": [(np.array([15, 70, 70]), np.array([35, 255, 255]))],
        }
        # Create masks and calculate percentages
        percentages = {}
        for color, ranges in color_ranges.items():
            mask = self.detect_color_regions(hsv, ranges)
            percentages[color] = self.calculate_percentage(mask, total_pixels)

        # Determine sign type
        if sign_type == "keep_right" and percentages["blue"] > 0.2 and percentages["white"] > 0 and percentages["red"] == 0:
            print("Detected Compulsory Keep Right sign!")
            return True
        elif sign_type == "no_left_turn" and percentages["red"] > 0.2 and percentages["white"] > 0.3 and percentages["black"] > 0.05:
            print("Detected No Left Turn sign!")
            return True
        elif sign_type == "no_parking" and percentages["blue"] > 0.2 and 0.03 < percentages["red"] < 0.37:
            print("Detected No Parking sign!")
            return True
        elif sign_type == "no_stopping" and percentages["blue"] > 0.2 and percentages["red"] > 0.37 and percentages["white"] <= 0:
            print("Detected No Stopping sign!")
            return True
        elif sign_type == "slow_down" and percentages["red"] > 0.1 and percentages["yellow"] > 0.3 and percentages["black"] > 0.39:
            print("Detected Slow Down sign!")
            return True
        elif sign_type == "children_crossing" and percentages["red"] > 0.1 and percentages["yellow_2"] > 0.3 and percentages["black"] < 0.39:
            print("Detected Children Crossing sign!")
            return True

        return False
        
    def is_triangle(self, contour):
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0:
            return False

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 3 or not (30 < area < 350000):
            return False

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if not (0.5 <= aspect_ratio <= 1.5): 
            return False

        # Calculate angles
        angles = []
        for i in range(3):
            p1, p2, p3 = approx[i], approx[(i + 1) % 3], approx[(i + 2) % 3]
            angle = self.calculate_angle(p1, p2, p3)
            angles.append(angle)
        
        # Check if all angles are within a reasonable triangle range
        if all(30 <= angle <= 120 for angle in angles):
            return True

        return False

    def calculate_angle(self, p1, p2, p3):
        v1 = np.array([p1[0][0] - p2[0][0], p1[0][1] - p2[0][1]])
        v2 = np.array([p3[0][0] - p2[0][0], p3[0][1] - p2[0][1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(cos_angle))
    

    def add_content_of_sign(self, frame, x, y, w, h, name):
        # Draw the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Determine the text position
        text_position = (x, y - 10 if y - 10 > 0 else y + 20)

        # Add the label text
        cv2.putText(frame, name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    def contains_triangle_sign_colors(self, roi):
        # Convert ROI to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define HSV ranges for colors
        color_ranges = {
            "red": [(np.array([0, 90, 50]), np.array([10, 255, 255])),
                    (np.array([160, 90, 50]), np.array([180, 255, 255]))],
            "yellow": [(np.array([15, 70, 70]), np.array([35, 255, 255]))],
            "black": [(np.array([0, 0, 0]), np.array([180, 255, 80]))],
        }
        
        # Calculate total pixels and threshold
        total_pixels = roi.shape[0] * roi.shape[1]
        threshold = total_pixels * 0.1  # 10% of the ROI

        # Create masks and check pixel count
        for color, ranges in color_ranges.items():
            if color == "red":
                mask = cv2.bitwise_or(cv2.inRange(hsv_roi, *ranges[0]),
                                    cv2.inRange(hsv_roi, *ranges[1]))
            else:
                mask = cv2.inRange(hsv_roi, *ranges[0])
            
            if cv2.countNonZero(mask) > threshold:
                return True

        return False

# Use the class with multiple videos
input_videos = ['video1.mp4', 'video2.mp4']
output_dir = "output_videos"
output_frame_dir = "output_frames"
color_ranges = {
    'red': [(0, 90, 50), (10, 255, 255), (160, 90, 50), (180, 255, 255)],
    'blue': [(100, 150, 0), (140, 255, 255)]
}

save_times = {
    'video1.mp4': [8000, 18500, 42500, 48000, 68000, 83700, 91000],
    'video2.mp4': [23200, 57000, 61000]
}

# Instantiate and process videos
tsr = TSR(input_videos, output_dir, output_frame_dir, color_ranges, save_times)
tsr.process_videos()
