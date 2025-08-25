#!/usr/bin/env python3

import rospy
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import logging
import os


class VisionDetection:
    
    def __init__(self):
        self.bridge = CvBridge()
        self.setup_vision()
        
    def setup_vision(self):
        """Setup YOLO key detection model"""
        try:
            rospy.loginfo("🔧 Setting up YOLO key detection...")
            
            # Use the new best.pt model for key detection
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(current_dir, '../src/best.pt')
            
            # Check if model exists
            if not os.path.exists(self.model_path):
                rospy.logwarn(f"YOLO model not found at {self.model_path}")
                rospy.logwarn("Please ensure best.pt is in the models directory")
                self.model = None
                self.use_yolo = False
            else:
                # Load YOLO model
                rospy.loginfo(f"Loading YOLO key detection model from: {self.model_path}")
                self.model = YOLO(self.model_path)
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    rospy.loginfo("🚀 Using GPU acceleration")
                else:
                    rospy.loginfo("🔄 Using CPU inference")
                
                self.use_yolo = True
                
                # Set detection parameters
                self.detection_confidence = 0.5
                self.iou_threshold = 0.45
                
                # Print available classes
                rospy.loginfo(f"Model classes: {list(self.model.names.values())}")
                
                # Suppress ultralytics logging
                logging.getLogger("ultralytics").setLevel(logging.ERROR)
                
                rospy.loginfo("✅ YOLO key detection model loaded successfully")
            
            # Initialize detection state
            self.last_detection_time = rospy.Time.now()
            self.detection_timeout = 5.0  # seconds
            
        except Exception as e:
            rospy.logerr(f"Vision setup error: {e}")
            self.model = None
            self.use_yolo = False

    def detect_keyboard_keys(self, image):
        """Detect individual keys using YOLO model"""
        keypoints_2d = {}
        keyboard_bbox = []
        
        try:
            if self.use_yolo and self.model is not None:
                rospy.logdebug("Running YOLO key detection...")
                
                # Run YOLO inference on individual keys
                results = self.model(
                    image, 
                    conf=self.detection_confidence,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                if len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        
                        rospy.loginfo_throttle(3, f"YOLO detected {len(boxes)} keys")
                        
                        detected_keys_list = []
                        
                        # Process each detected key
                        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                            
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box
                            confidence = float(conf)
                            class_id = int(cls)
                            
                            # Get class name from model
                            if class_id < len(self.model.names):
                                class_name = self.model.names[class_id]
                            else:
                                continue
                            
                            # Calculate center point of detected key
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            # Map detected class to usable key name
                            key_name = self.normalize_key_name(class_name)
                            
                            if key_name and confidence > self.detection_confidence:
                                keypoints_2d[key_name] = [center_x, center_y]
                                detected_keys_list.append(f"{key_name}({confidence:.2f})")
                        
                        # Print detected keys periodically
                        # if detected_keys_list:
                            # rospy.loginfo_throttle(2, f"🔍 Detected keys: {detected_keys_list}")
                        
                        # Calculate overall keyboard region
                        if keypoints_2d:
                            all_x = [pos[0] for pos in keypoints_2d.values()]
                            all_y = [pos[1] for pos in keypoints_2d.values()]
                            
                            margin = 50
                            x1 = max(0, min(all_x) - margin)
                            y1 = max(0, min(all_y) - margin) 
                            x2 = min(image.shape[1], max(all_x) + margin)
                            y2 = min(image.shape[0], max(all_y) + margin)
                            
                            keyboard_bbox = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                            self.last_detection_time = rospy.Time.now()
            
            else:
                rospy.logwarn_throttle(10, "YOLO model not available")
                
        except Exception as e:
            rospy.logerr(f"YOLO key detection error: {e}")

        return keypoints_2d, keyboard_bbox
    
    def normalize_key_name(self, detected_class):
        """Normalize detected class name to standard key name"""
        
        # Convert to uppercase for consistency
        key_upper = detected_class.upper().strip()
        
        # Handle common key name variations
        key_mappings = {
            # Letters (should be direct)
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G',
            'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 
            'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
            'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
            
            # Numbers
            '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
            '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
            
            # Special keys - handle variations
            'SPACE': 'SPACE', 'SPACEBAR': 'SPACE', 'SPACE_BAR': 'SPACE',
            'ENTER': 'ENTER', 'RETURN': 'ENTER',
            'BACKSPACE': 'BACKSPACE', 'DELETE': 'BACKSPACE', 'DEL': 'BACKSPACE',
            'SHIFT': 'SHIFT', 'SHIFT_L': 'SHIFT', 'SHIFT_R': 'SHIFT',
            'CTRL': 'CTRL', 'CONTROL': 'CTRL',
            'ALT': 'ALT',
            'TAB': 'TAB',
            'ESC': 'ESC', 'ESCAPE': 'ESC'
        }
        
        # Direct mapping
        if key_upper in key_mappings:
            return key_mappings[key_upper]
        
        # Single character keys
        if len(key_upper) == 1 and (key_upper.isalpha() or key_upper.isdigit()):
            return key_upper
        
        # Log unknown keys for debugging
        rospy.logwarn_throttle(30, f"Unknown key detected: '{detected_class}' -> normalized to '{key_upper}'")
        return key_upper

    def publish_annotated_image(self, color_image, keypoints_2d, keyboard_bbox, keypoints_3d, session_coordinates, camera_frame, use_yolo, annotated_image_pub):
        """Enhanced detection visualization"""
        if color_image is None:
            return
            
        try:
            annotated = color_image.copy()
            
            # Draw keyboard bounding box if available
            if keyboard_bbox:
                pts = np.array(keyboard_bbox, np.int32)
                cv2.polylines(annotated, [pts], True, (255, 0, 0), 3)
                cv2.putText(annotated, "Keyboard Region", 
                           (int(pts[0][0]), int(pts[0][1] - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Draw detected keys
            for key, (x, y) in keypoints_2d.items():
                # Color coding: 
                # Green = fresh detection with good depth
                # Yellow = session coordinates available
                # Red = no 3D data available
                
                if key in keypoints_3d:
                    # Fresh detection
                    depth_data = keypoints_3d[key]
                    if isinstance(depth_data, dict) and 'depth' in depth_data:
                        if 0.1 < depth_data['depth'] < 2.0:
                            color = (0, 255, 0)  # Green - good fresh detection
                            status = "FRESH+GOOD"
                        else:
                            color = (255, 0, 0)  # Red - poor depth
                            status = "FRESH+POOR"
                    else:
                        color = (0, 0, 255)  # Red - no depth info
                        status = "FRESH+NO_DEPTH"
                elif key in session_coordinates:
                    color = (0, 255, 255)  # Yellow - session coordinates available
                    status = "SESSION"
                else:
                    color = (0, 0, 255)  # Red - no 3D data
                    status = "NO_3D"
                
                # Draw key center
                cv2.circle(annotated, (int(x), int(y)), 10, color, -1)
                cv2.circle(annotated, (int(x), int(y)), 12, (255, 255, 255), 2)
                
                # Draw key label
                label = f"{key} ({status})"
                cv2.putText(annotated, label, 
                           (int(x) - 30, int(y) - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Status information
            status_lines = [
                f"YOLO Model: {'Active' if use_yolo else 'Inactive'}",
                f"2D Keys Detected: {len(keypoints_2d)}",
                f"3D Keys (Fresh): {len(keypoints_3d)}",
                f"Session Total: {len(session_coordinates)}",
                f"Transform Frame: {camera_frame}"
            ]
            
            for i, line in enumerate(status_lines):
                y_pos = 30 + i * 25
                cv2.putText(annotated, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Publish debug image
            msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            annotated_image_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Debug image error: {e}")