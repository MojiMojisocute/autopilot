import cv2
import numpy as np
from ultralytics import YOLO
import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import threading
from enum import Enum
import os
from sklearn.cluster import KMeans
from scipy.spatial import distance

class TrafficLightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"

class DrivingState(Enum):
    NORMAL = "normal"
    STOPPING = "stopping"
    OVERTAKING = "overtaking"
    FOLLOWING = "following"
    EMERGENCY_BRAKE = "emergency_brake"

class VehicleLightType(Enum):
    BRAKE = "brake"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    NONE = "none"

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

@dataclass
class VehicleState:
    speed: float = 0.0
    position: Tuple[float, float] = (0.0, 0.0)
    lane: int = 1
    target_lane: int = 1
    driving_state: DrivingState = DrivingState.NORMAL

@dataclass
class LaneInfo:
    left_line: Optional[Tuple[int, int, int, int]] = None
    right_line: Optional[Tuple[int, int, int, int]] = None
    center_line: Optional[Tuple[int, int, int, int]] = None
    lane_width: float = 0.0
    lane_position: float = 0.0
    
class AutopilotCar:
    def __init__(self, model_name="yolov8n.pt", input_source=0, output_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            self.model = YOLO(model_name)
            print(f"Model {model_name} loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to download model...")
            self.model = YOLO(model_name)

        self.vehicle_state = VehicleState()
        self.traffic_light_state = TrafficLightState.UNKNOWN
        self.last_traffic_light_check = 0
        self.lane_width = 50
        self.lane_center = 100 
        self.min_following_distance = 100
        self.max_speed = 90
        self.current_speed = 0
        self.prev_lane_info = None
        self.traffic_light_history = []
        self.brake_light_history = {}

        self.target_classes = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck",
            9: "traffic_light",
            11: "stop_sign",
            12: "parking_meter"
        }
        
        self.input_source = input_source
        self.output_path = output_path
        self.cap = cv2.VideoCapture(input_source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {input_source}")
            
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        self.frame_count = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(input_source, str) else 0
        self.processing_times = []
        
        print(f"Input source: {input_source}")
        print(f"Video properties: {self.width}x{self.height} @ {self.fps} fps")
        if self.total_frames > 0:
            print(f"Total frames: {self.total_frames}")
        if output_path:
            print(f"Output will be saved to: {output_path}")
        print("Autopilot system initialized")
    
    def detect_objects(self, frame):
        results = self.model(frame, conf=0.5, verbose=False)
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id in self.target_classes:
                        detection = Detection(
                            x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                            confidence=float(confidence),
                            class_id=class_id,
                            class_name=self.target_classes[class_id]
                        )
                        detections.append(detection)
        return detections
    
    def detect_lanes_improved(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blur)
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
        height, width = edges.shape
        roi = np.zeros_like(edges)
        roi_vertices = np.array([[
            [int(width * 0.1), height],
            [int(width * 0.4), int(height * 0.6)],
            [int(width * 0.6), int(height * 0.6)],
            [int(width * 0.9), height]
        ]], dtype=np.int32)
        
        cv2.fillPoly(roi, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, roi)
        lines = cv2.HoughLinesP(
            masked_edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=100,
            maxLineGap=50
        )
        
        if lines is None:
            return LaneInfo()
        
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:  
                continue
            if slope < 0 and x1 < width // 2: 
                left_lines.append((x1, y1, x2, y2, slope))
            elif slope > 0 and x1 > width // 2: 
                right_lines.append((x1, y1, x2, y2, slope))
        def average_lines(lines):
            if not lines:
                return None
            x_coords = []
            y_coords = []
            weights = []
            
            for x1, y1, x2, y2, slope in lines:
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                weights.extend([length, length])
                x_coords.extend([x1, x2])
                y_coords.extend([y1, y2])
            
            if len(x_coords) < 2:
                return None
            
            A = np.vstack([x_coords, np.ones(len(x_coords))]).T
            W = np.diag(weights)
            coeffs = np.linalg.lstsq(W @ A, W @ np.array(y_coords), rcond=None)[0]
            
            slope, intercept = coeffs
            
            y1 = height
            y2 = int(height * 0.6)
            x1 = int((y1 - intercept) / slope) if slope != 0 else 0
            x2 = int((y2 - intercept) / slope) if slope != 0 else 0
            
            return (x1, y1, x2, y2)
        
        left_line = average_lines(left_lines)
        right_line = average_lines(right_lines)
        
        center_line = None
        lane_width = 0
        lane_position = 0
        
        if left_line and right_line:
            # Calculate center line
            lx1, ly1, lx2, ly2 = left_line
            rx1, ry1, rx2, ry2 = right_line
            
            cx1 = (lx1 + rx1) // 2
            cy1 = (ly1 + ry1) // 2
            cx2 = (lx2 + rx2) // 2
            cy2 = (ly2 + ry2) // 2
            
            center_line = (cx1, cy1, cx2, cy2)
            lane_width = abs(rx1 - lx1)
            
            vehicle_x = width // 2
            lane_position = (vehicle_x - cx1) / (lane_width / 2) if lane_width > 0 else 0
        
        return LaneInfo(
            left_line=left_line,
            right_line=right_line,
            center_line=center_line,
            lane_width=lane_width,
            lane_position=lane_position
        )
    
    def detect_traffic_light_color_improved(self, detection, frame):
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2

        margin = 5
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(frame.shape[1], x2 + margin)
        y2 = min(frame.shape[0], y2 + margin)
        
        traffic_light_roi = frame[y1:y2, x1:x2]
        
        if traffic_light_roi.size == 0:
            return TrafficLightState.UNKNOWN

        height, width = traffic_light_roi.shape[:2]
        if height < 50 or width < 30:
            traffic_light_roi = cv2.resize(traffic_light_roi, (30, 90))

        hsv = cv2.cvtColor(traffic_light_roi, cv2.COLOR_BGR2HSV)

        red_lower1 = np.array([0, 120, 70])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 120, 70])
        red_upper2 = np.array([180, 255, 255])
        
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        
        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        total_pixels = traffic_light_roi.shape[0] * traffic_light_roi.shape[1]
        
        threshold = max(50, total_pixels * 0.05)
        
        if red_pixels > threshold and red_pixels > yellow_pixels and red_pixels > green_pixels:
            return TrafficLightState.RED
        elif yellow_pixels > threshold and yellow_pixels > green_pixels:
            return TrafficLightState.YELLOW
        elif green_pixels > threshold:
            return TrafficLightState.GREEN
        else:
            return TrafficLightState.UNKNOWN
    
    def detect_vehicle_lights_improved(self, detection, frame):
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        
        vehicle_height = y2 - y1
        vehicle_width = x2 - x1
        
        rear_y1 = y1 + int(vehicle_height * 0.6)
        rear_region = frame[rear_y1:y2, x1:x2]
        
        if rear_region.size == 0:
            return VehicleLightType.NONE
        
        hsv = cv2.cvtColor(rear_region, cv2.COLOR_BGR2HSV)
        
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        orange_lower = np.array([10, 100, 100])
        orange_upper = np.array([25, 255, 255])
        orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)

        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        red_pixels = cv2.countNonZero(red_mask)
        orange_pixels = cv2.countNonZero(orange_mask)
        total_pixels = rear_region.shape[0] * rear_region.shape[1]
        
        if red_pixels > total_pixels * 0.05:
            if len(red_contours) >= 2:
                centroids = []
                for contour in red_contours:
                    if cv2.contourArea(contour) > 20:
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            centroids.append(cx)
                
                if len(centroids) >= 2:
                    center_x = vehicle_width // 2
                    left_lights = [c for c in centroids if c < center_x]
                    right_lights = [c for c in centroids if c > center_x]
                    
                    if len(left_lights) > 0 and len(right_lights) > 0:
                        return VehicleLightType.BRAKE

        if orange_pixels > total_pixels * 0.03:
            if len(orange_contours) > 0:
                largest_contour = max(orange_contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    center_x = vehicle_width // 2
                    
                    if cx < center_x * 0.7:  # Left side
                        return VehicleLightType.TURN_LEFT
                    elif cx > center_x * 1.3:  # Right side
                        return VehicleLightType.TURN_RIGHT
        
        return VehicleLightType.NONE
    
    def calculate_following_distance(self, vehicle_detection):
        box_height = vehicle_detection.y2 - vehicle_detection.y1
        box_width = vehicle_detection.x2 - vehicle_detection.x1
        
        if box_height > 0 and box_width > 0:
            area = box_height * box_width
            estimated_distance = max(10, min(500, 10000 / area))
            return estimated_distance
        
        return 200 
    
    def plan_driving_action(self, detections, lane_info, frame):
        current_time = time.time()

        traffic_lights = [d for d in detections if d.class_name == "traffic_light"]
        if traffic_lights:
            closest_light = max(traffic_lights, key=lambda x: (x.x2 - x.x1) * (x.y2 - x.y1))
            light_state = self.detect_traffic_light_color_improved(closest_light, frame)

            self.traffic_light_history.append(light_state)
            if len(self.traffic_light_history) > 5:
                self.traffic_light_history.pop(0)

            if self.traffic_light_history:
                from collections import Counter
                state_counts = Counter(self.traffic_light_history)
                self.traffic_light_state = state_counts.most_common(1)[0][0]
        
        vehicles_ahead = [d for d in detections if d.class_name in ["car", "truck", "bus", "motorcycle"] 
                         and d.y2 > frame.shape[0] // 2]

        pedestrians = [d for d in detections if d.class_name == "person"]
        stop_signs = [d for d in detections if d.class_name == "stop_sign"]

        if pedestrians:
            self.vehicle_state.driving_state = DrivingState.EMERGENCY_BRAKE
            self.current_speed = max(0, self.current_speed - 15)
            return

        if self.traffic_light_state == TrafficLightState.RED or stop_signs:
            self.vehicle_state.driving_state = DrivingState.STOPPING
            self.current_speed = max(0, self.current_speed - 8)
            return

        if self.traffic_light_state == TrafficLightState.YELLOW:
            self.vehicle_state.driving_state = DrivingState.STOPPING
            self.current_speed = max(15, self.current_speed - 3)
            return

        if vehicles_ahead:
            closest_vehicle = min(vehicles_ahead, key=lambda x: x.y2)
            distance = self.calculate_following_distance(closest_vehicle)
            
            light_type = self.detect_vehicle_lights_improved(closest_vehicle, frame)

            if light_type == VehicleLightType.BRAKE:
                self.vehicle_state.driving_state = DrivingState.EMERGENCY_BRAKE
                self.current_speed = max(0, self.current_speed - 12)
            elif distance < self.min_following_distance:
                self.vehicle_state.driving_state = DrivingState.FOLLOWING
                self.current_speed = max(10, self.current_speed - 5)
            else:
                if (distance < self.min_following_distance * 2 and 
                    self.can_overtake(detections) and 
                    lane_info.lane_width > 0):
                    self.vehicle_state.driving_state = DrivingState.OVERTAKING
                    self.current_speed = min(self.max_speed, self.current_speed + 3)
                else:
                    self.vehicle_state.driving_state = DrivingState.FOLLOWING
                    self.current_speed = max(15, self.current_speed - 1)

        else:
            self.vehicle_state.driving_state = DrivingState.NORMAL
            if self.traffic_light_state == TrafficLightState.GREEN:
                self.current_speed = min(self.max_speed, self.current_speed + 2)
            else:
                self.current_speed = min(self.max_speed, self.current_speed + 1)
    
    def can_overtake(self, detections):
        oncoming_vehicles = [d for d in detections if d.class_name in ["car", "truck", "bus", "motorcycle"] 
                           and d.y1 < self.height // 3]
        return len(oncoming_vehicles) == 0
    
    def draw_visualizations(self, frame, detections, lane_info):
        for detection in detections:
            x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2

            if detection.class_name == "traffic_light":
                color = (0, 255, 255)  # Yellow
            elif detection.class_name in ["car", "truck", "bus", "motorcycle"]:
                color = (255, 0, 0)  # Blue
            elif detection.class_name == "stop_sign":
                color = (0, 0, 255)  # Red
            elif detection.class_name == "person":
                color = (0, 255, 0)  # Green
            else:
                color = (128, 128, 128)  # Gray

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if detection.class_name in ["car", "truck", "bus", "motorcycle"]:
                light_type = self.detect_vehicle_lights_improved(detection, frame)
                if light_type == VehicleLightType.BRAKE:
                    cv2.putText(frame, "BRAKE!", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif light_type == VehicleLightType.TURN_LEFT:
                    cv2.putText(frame, "LEFT", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                elif light_type == VehicleLightType.TURN_RIGHT:
                    cv2.putText(frame, "RIGHT", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        if lane_info.left_line:
            x1, y1, x2, y2 = lane_info.left_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        if lane_info.right_line:
            x1, y1, x2, y2 = lane_info.right_line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        if lane_info.center_line:
            x1, y1, x2, y2 = lane_info.center_line
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

        height, width = frame.shape[:2]
        roi_vertices = np.array([[
            [int(width * 0.1), height],
            [int(width * 0.4), int(height * 0.6)],
            [int(width * 0.6), int(height * 0.6)],
            [int(width * 0.9), height]
        ]], dtype=np.int32)
        cv2.polylines(frame, [roi_vertices], True, (100, 100, 100), 1)

        status_text = [
            f"Speed: {self.current_speed:.1f} km/h",
            f"State: {self.vehicle_state.driving_state.value}",
            f"Traffic Light: {self.traffic_light_state.value}",
            f"Lane Position: {lane_info.lane_position:.2f}",
            f"Lane Width: {lane_info.lane_width:.1f}px",
            f"Device: {self.device}"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        self.draw_speed_gauge(frame)
        
        return frame
    
    def draw_speed_gauge(self, frame):
        gauge_center = (frame.shape[1] - 100, 100)
        gauge_radius = 50
        
        cv2.circle(frame, gauge_center, gauge_radius, (255, 255, 255), 2)
        
        speed_angle = (self.current_speed / self.max_speed) * 180 - 90
        needle_x = int(gauge_center[0] + (gauge_radius - 10) * np.cos(np.radians(speed_angle)))
        needle_y = int(gauge_center[1] + (gauge_radius - 10) * np.sin(np.radians(speed_angle)))
        cv2.line(frame, gauge_center, (needle_x, needle_y), (0, 255, 0), 3)

        cv2.putText(frame, "0", (gauge_center[0] - 60, gauge_center[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, str(self.max_speed), (gauge_center[0] + 40, gauge_center[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        print("Starting improved autopilot system...")
        if isinstance(self.input_source, str):
            print(f"Processing video: {self.input_source}")
            print("Press 'q' to quit, 'p' to pause/resume")
        else:
            print("Using camera input")
            print("Press 'q' to quit, 's' to save screenshot")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.input_source, str):
                        print("Reached end of video")
                    else:
                        print("Failed to capture frame")
                    break
                
                self.frame_count += 1
                start_time = time.time()

                detections = self.detect_objects(frame)
                lane_info = self.detect_lanes_improved(frame)
                self.plan_driving_action(detections, lane_info, frame)
                frame = self.draw_visualizations(frame, detections, lane_info)

                if self.total_frames > 0:
                    progress = (self.frame_count / self.total_frames) * 100
                    progress_text = f"Progress: {progress:.1f}% ({self.frame_count}/{self.total_frames})"
                    cv2.putText(frame, progress_text, (10, frame.shape[0] - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    bar_width = 300
                    bar_height = 10
                    bar_x = frame.shape[1] - bar_width - 10
                    bar_y = frame.shape[0] - 30
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress / 100), bar_y + bar_height), (0, 255, 0), -1)

                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                fps_text = f"FPS: {1/processing_time:.1f}"
                cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if self.video_writer:
                    self.video_writer.write(frame)

            cv2.imshow("Improved Autopilot Car", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('s') and not isinstance(self.input_source, str):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                screenshot_path = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(screenshot_path, frame)
                print(f"Screenshot saved: {screenshot_path}")
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            avg_fps = 1 / avg_time
            print(f"\nProcessing Statistics:")
            print(f"Frames processed: {self.frame_count}")
            print(f"Average processing time: {avg_time:.3f}s")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Min processing time: {min(self.processing_times):.3f}s")
            print(f"Max processing time: {max(self.processing_times):.3f}s")
        
        if self.output_path:
            print(f"Output video saved to: {self.output_path}")
        
        print("Improved autopilot system stopped")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved Autopilot Car with YOLOv8')
    parser.add_argument('--input', '-i', type=str, default=0, 
                       help='Input source: 0 for camera, or path to video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--model', '-m', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='YOLOv8 model size')
    
    args = parser.parse_args()

    input_source = int(args.input) if args.input.isdigit() else args.input
    
    try:
        autopilot = AutopilotCar(
            model_name=args.model,
            input_source=input_source,
            output_path=args.output
        )
        autopilot.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_video_batch(video_folder, output_folder, model_name="yolov8n.pt"):
    import glob
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))
    
    for video_file in video_files:
        print(f"\nProcessing: {video_file}")
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        output_path = os.path.join(output_folder, f"{video_name}_improved_autopilot.mp4")
        
        try:
            autopilot = AutopilotCar(model_name, video_file, output_path)
            autopilot.run()
        except Exception as e:
            print(f"Error processing {video_file}: {e}")

def create_test_report(video_path, output_folder):
    import json
    
    print(f"\nCreating test report for: {video_path}")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt']
    results = {}
    
    for model in models:
        print(f"Testing with {model}...")
        output_path = os.path.join(output_folder, f"{video_name}_{model.replace('.pt', '')}_improved.mp4")
        
        try:
            autopilot = AutopilotCar(model, video_path, output_path)
            start_time = time.time()
            autopilot.run()
            total_time = time.time() - start_time
            
            results[model] = {
                'total_time': total_time,
                'frames_processed': autopilot.frame_count,
                'avg_fps': autopilot.frame_count / total_time if total_time > 0 else 0,
                'avg_processing_time': np.mean(autopilot.processing_times) if autopilot.processing_times else 0,
                'min_processing_time': min(autopilot.processing_times) if autopilot.processing_times else 0,
                'max_processing_time': max(autopilot.processing_times) if autopilot.processing_times else 0
            }
            
        except Exception as e:
            results[model] = {'error': str(e)}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    report_path = os.path.join(output_folder, f"{video_name}_improved_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Test report saved to: {report_path}")
    
    print("\nPerformance Summary:")
    for model, result in results.items():
        if 'error' not in result:
            print(f"{model}: {result['avg_fps']:.1f} FPS, {result['avg_processing_time']:.3f}s avg")
        else:
            print(f"{model}: Error - {result['error']}")
    
    return results

def demo_lane_detection(image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not load image: {image_path}")
        return
    
    autopilot = AutopilotCar()
    
    lane_info = autopilot.detect_lanes_improved(frame)
    
    if lane_info.left_line:
        x1, y1, x2, y2 = lane_info.left_line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    if lane_info.right_line:
        x1, y1, x2, y2 = lane_info.right_line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    if lane_info.center_line:
        x1, y1, x2, y2 = lane_info.center_line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
    cv2.putText(frame, f"Lane Width: {lane_info.lane_width:.1f}px", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Lane Position: {lane_info.lane_position:.2f}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Lane Detection Demo", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = f"lane_detection_result_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, frame)
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    main()