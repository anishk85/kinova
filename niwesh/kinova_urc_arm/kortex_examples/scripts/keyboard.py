#!/usr/bin/env python3

import rospy
import logging
import os
import json
import numpy as np
import torch
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped, PointStamped
from sensor_msgs.msg import Image, CameraInfo, JointState
from cv_bridge import CvBridge
from tf2_geometry_msgs import do_transform_point
import tf2_ros
import cv2
from ultralytics import YOLO
import threading
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import copy


class RoboticTypingController:
    
    def __init__(self):
        rospy.init_node("robotic_typing_controller", log_level=rospy.INFO)
        logging.basicConfig(level=logging.INFO)

        rospy.loginfo("🤖 Initializing Robotic Typing Controller...")

        self.robot_namespace = "my_gen3"  
        self.base_frame = 'base_link'
        self.camera_frame = 'camera_link'

        # Camera topics
        self.depth_topic = '/camera/depth/image_rect_raw'
        self.color_topic = '/camera/color/image_raw'
        self.camera_info_topic = '/camera/color/camera_info'

        # Typing parameters - More conservative values
        self.press_depth = 0.002  # Reduced from 0.003
        self.dwell_time = 0.3     # Increased from 0.2
        self.approach_height = 0.015  # Reduced from 0.02
        self.safe_height = 0.04   # Reduced from 0.05
        
        # Movement parameters
        self.max_velocity_scale = 0.2  # Very conservative
        self.max_acceleration_scale = 0.1  # Very conservative
        
        self.setup_vision()
        self.load_keyboard_layout()
        self.setup_moveit()

        self.bridge = CvBridge()
        self.setup_subscribers_publishers()

        # State variables
        self.keypoints_3d = {}
        self.typing_active = False
        self.keyboard_detected = False
        self.depth_image = None
        self.color_image = None
        self.camera_info = None
        self.current_joint_state = None
        self.detection_region_expanded = False
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Home position for efficient movement
        self.home_position = None
        self.last_key_position = None

        rospy.loginfo("✅ Robotic Typing Controller initialized successfully!")

    def setup_vision(self):
        """Setup YOLO key detection model with improved parameters"""
        try:
            rospy.loginfo("🔧 Setting up YOLO key detection...")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.model_path = os.path.join(current_dir, '../src/best.pt')
            
            if not os.path.exists(self.model_path):
                rospy.logwarn(f"YOLO model not found at {self.model_path}")
                self.model = None
                self.use_yolo = False
            else:
                rospy.loginfo(f"Loading YOLO key detection model from: {self.model_path}")
                self.model = YOLO(self.model_path)
                
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    rospy.loginfo("🚀 Using GPU acceleration")
                else:
                    rospy.loginfo("🔄 Using CPU inference")
                
                self.use_yolo = True
                
                # More aggressive detection parameters for better coverage
                self.detection_confidence = 0.3  # Reduced from 0.5
                self.iou_threshold = 0.3         # Reduced from 0.45
                
                rospy.loginfo(f"Model classes: {list(self.model.names.values())}")
                
                # Suppress ultralytics logging
                import logging
                logging.getLogger("ultralytics").setLevel(logging.ERROR)
                
                rospy.loginfo("✅ YOLO key detection model loaded successfully")
            
            self.last_detection_time = rospy.Time.now()
            self.detection_timeout = 5.0
            
        except Exception as e:
            rospy.logerr(f"Vision setup error: {e}")
            self.model = None
            self.use_yolo = False

    def load_keyboard_layout(self):
        """Initialize key detection with fallback layout"""
        try:
            rospy.loginfo("🔧 Initializing key detection system...")
            
            self.detected_keys = {}
            
            # Fallback QWERTY layout for missing keys (relative positions)
            self.fallback_layout = {
                # Top row numbers
                '1': (0.05, 0.1), '2': (0.15, 0.1), '3': (0.25, 0.1), '4': (0.35, 0.1), 
                '5': (0.45, 0.1), '6': (0.55, 0.1), '7': (0.65, 0.1), '8': (0.75, 0.1), 
                '9': (0.85, 0.1), '0': (0.95, 0.1),
                
                # First letter row
                'Q': (0.1, 0.3), 'W': (0.2, 0.3), 'E': (0.3, 0.3), 'R': (0.4, 0.3), 
                'T': (0.5, 0.3), 'Y': (0.6, 0.3), 'U': (0.7, 0.3), 'I': (0.8, 0.3), 
                'O': (0.9, 0.3), 'P': (1.0, 0.3),
                
                # Second letter row
                'A': (0.15, 0.5), 'S': (0.25, 0.5), 'D': (0.35, 0.5), 'F': (0.45, 0.5), 
                'G': (0.55, 0.5), 'H': (0.65, 0.5), 'J': (0.75, 0.5), 'K': (0.85, 0.5), 
                'L': (0.95, 0.5),
                
                # Third letter row
                'Z': (0.2, 0.7), 'X': (0.3, 0.7), 'C': (0.4, 0.7), 'V': (0.5, 0.7), 
                'B': (0.6, 0.7), 'N': (0.7, 0.7), 'M': (0.8, 0.7),
                
                # Space bar
                'SPACE': (0.5, 0.9)
            }
            
            rospy.loginfo("✅ Key detection system ready with fallback layout")
                
        except Exception as e:
            rospy.logerr(f"Error initializing key detection: {e}")

    def setup_moveit(self):
        """Initialize MoveIt components with conservative parameters"""
        try:
            rospy.loginfo("🔧 Setting up MoveIt...")
            
            moveit_commander.roscpp_initialize([])
            
            self.robot = moveit_commander.RobotCommander(
                robot_description=f"{self.robot_namespace}/robot_description",
                ns=self.robot_namespace
            )
            self.scene = moveit_commander.PlanningSceneInterface(ns=self.robot_namespace)

            available_groups = self.robot.get_group_names()
            rospy.loginfo(f"Available planning groups: {available_groups}")
            
            self.move_group = moveit_commander.MoveGroupCommander(
                "arm",
                robot_description=f"{self.robot_namespace}/robot_description",
                ns=self.robot_namespace
            )
            
            self.configure_moveit_for_typing()
            
            # Store current position as home
            try:
                self.home_position = self.move_group.get_current_pose().pose
                rospy.loginfo("📍 Home position stored")
            except:
                rospy.logwarn("Could not store home position")
            
            rospy.loginfo("✅ MoveIt setup complete.")
            rospy.loginfo(f"   Planning Group: arm")
            rospy.loginfo(f"   Planning Frame: {self.move_group.get_planning_frame()}")
            rospy.loginfo(f"   End Effector: {self.move_group.get_end_effector_link()}")

        except Exception as e:
            rospy.logfatal(f"❌ MoveIt setup failed: {e}")
            raise

    def configure_moveit_for_typing(self):
        """Configure MoveIt parameters for precise, safe typing"""
        # Increased planning time for better solutions
        self.move_group.set_planning_time(15.0)
        self.move_group.set_num_planning_attempts(20)
        self.move_group.allow_replanning(True)
        
        # Very conservative velocity and acceleration scaling
        self.move_group.set_max_velocity_scaling_factor(self.max_velocity_scale)
        self.move_group.set_max_acceleration_scaling_factor(self.max_acceleration_scale)
        
        # Tight tolerances for precision
        self.move_group.set_goal_position_tolerance(0.001)  # 1mm
        self.move_group.set_goal_orientation_tolerance(0.05)
        
        # Use RRTConnect which is generally more reliable
        self.move_group.set_planner_id("RRTConnect")
        
        rospy.loginfo(f"MoveIt configured: vel={self.max_velocity_scale}, acc={self.max_acceleration_scale}")

    def setup_subscribers_publishers(self):
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        rospy.Subscriber(self.color_topic, Image, self.color_callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(f"/{self.robot_namespace}/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/type_text", String, self.type_text_callback)
        
        self.typing_status_pub = rospy.Publisher("/typing_status", String, queue_size=10)
        self.key_coordinates_pub = rospy.Publisher("/key_coordinates_3d", String, queue_size=10)
        self.annotated_image_pub = rospy.Publisher("/annotated_keyboard", Image, queue_size=10)
        self.trajectory_pub = rospy.Publisher(
            f"/{self.robot_namespace}/move_group/display_planned_path", 
            DisplayTrajectory, queue_size=10
        )

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as e:
            rospy.logerr(f"Depth callback error: {e}")

    def color_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.process_keyboard_detection()
        except Exception as e:
            rospy.logerr(f"Color callback error: {e}")

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def joint_state_callback(self, msg):
        self.current_joint_state = msg

    def publish_status(self, message):
        try:
            status_msg = String()
            status_msg.data = message
            self.typing_status_pub.publish(status_msg)
            rospy.loginfo(f"Status: {message}")
        except Exception as e:
            rospy.logerr(f"Status publish error: {e}")

    def process_keyboard_detection(self):
        """Enhanced keyboard detection with multi-scale processing"""
        if self.color_image is None or self.depth_image is None or self.camera_info is None:
            return

        try:
            # Multi-scale detection for better coverage
            keypoints_2d_primary, keyboard_bbox = self.detect_keyboard_keys(self.color_image)
            
            # If we don't have many keys, try with expanded region and lower confidence
            if len(keypoints_2d_primary) < 10 and not self.detection_region_expanded:
                rospy.loginfo("🔍 Expanding detection region for better coverage...")
                keypoints_2d_secondary = self.detect_keys_expanded_region(self.color_image)
                keypoints_2d_primary.update(keypoints_2d_secondary)
                self.detection_region_expanded = True
            
            # Apply fallback layout for common missing keys
            keypoints_2d_complete = self.apply_fallback_layout(keypoints_2d_primary, keyboard_bbox)
            
            self.publish_annotated_image(keypoints_2d_complete, keyboard_bbox)
            
            if not keypoints_2d_complete:
                time_since_detection = (rospy.Time.now() - self.last_detection_time).to_sec()
                if time_since_detection > self.detection_timeout:
                    self.keyboard_detected = False
                return

            # Enhanced 3D position calculation with depth filtering
            keypoints_3d_camera = self.calculate_3d_positions_enhanced(keypoints_2d_complete)
            
            if not keypoints_3d_camera:
                rospy.logwarn_throttle(5, "No valid 3D positions calculated")
                return

            # Transform and validate
            keypoints_3d_base = {}
            for key, pos_3d_cam in keypoints_3d_camera.items():
                pos_3d_base = self.transform_to_base_frame(pos_3d_cam)
                if pos_3d_base is not None and self.validate_workspace(pos_3d_base):
                    keypoints_3d_base[key] = {
                        'position': pos_3d_base.tolist(),
                        'depth': pos_3d_cam[2],
                        'pixel_coords': keypoints_2d_complete[key]
                    }

            if keypoints_3d_base:
                self.keypoints_3d = keypoints_3d_base
                self.keyboard_detected = True
                
                detection_data = {
                    "timestamp": rospy.Time.now().to_sec(),
                    "detection_method": "YOLO_enhanced_multi_scale",
                    "total_keys_detected": len(keypoints_3d_base),
                    "keys": keypoints_3d_base
                }
                
                self.key_coordinates_pub.publish(String(data=json.dumps(detection_data)))
                
                rospy.loginfo_throttle(5, f"✅ Detected {len(keypoints_3d_base)} keys with 3D coordinates")
                rospy.loginfo_throttle(10, f"📋 Available keys: {sorted(keypoints_3d_base.keys())}")

        except Exception as e:
            rospy.logerr(f"Key detection processing error: {e}")

    def detect_keys_expanded_region(self, image):
        """Detect keys with expanded parameters for better coverage"""
        keypoints_2d = {}
        
        try:
            if self.use_yolo and self.model is not None:
                # More aggressive detection parameters
                results = self.model(
                    image, 
                    conf=0.2,  # Lower confidence
                    iou=0.2,   # Lower IoU
                    verbose=False
                )
                
                if len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                            x1, y1, x2, y2 = box
                            confidence = float(conf)
                            class_id = int(cls)
                            
                            if class_id < len(self.model.names):
                                class_name = self.model.names[class_id]
                                key_name = self.normalize_key_name(class_name)
                                
                                if key_name and confidence > 0.2:  # Lower threshold
                                    center_x = int((x1 + x2) / 2)
                                    center_y = int((y1 + y2) / 2)
                                    keypoints_2d[key_name] = [center_x, center_y]
                
        except Exception as e:
            rospy.logerr(f"Expanded detection error: {e}")
            
        return keypoints_2d

    def apply_fallback_layout(self, detected_keys, keyboard_bbox):
        """Apply fallback layout for missing common keys"""
        complete_keypoints = detected_keys.copy()
        
        if not keyboard_bbox or len(detected_keys) < 5:
            return complete_keypoints
        
        try:
            # Calculate keyboard bounds
            all_x = [pos[0] for pos in detected_keys.values()]
            all_y = [pos[1] for pos in detected_keys.values()]
            
            if not all_x or not all_y:
                return complete_keypoints
            
            kb_left = min(all_x)
            kb_right = max(all_x)
            kb_top = min(all_y)
            kb_bottom = max(all_y)
            
            kb_width = kb_right - kb_left
            kb_height = kb_bottom - kb_top
            
            # Add missing essential keys using layout
            essential_keys = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 
                            'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
                            'Z', 'X', 'C', 'V', 'B', 'N', 'M',
                            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                            'SPACE']
            
            added_keys = []
            for key in essential_keys:
                if key not in complete_keypoints and key in self.fallback_layout:
                    rel_x, rel_y = self.fallback_layout[key]
                    abs_x = int(kb_left + rel_x * kb_width)
                    abs_y = int(kb_top + rel_y * kb_height)
                    
                    # Validate position is within image bounds
                    if (0 < abs_x < self.color_image.shape[1] and 
                        0 < abs_y < self.color_image.shape[0]):
                        complete_keypoints[key] = [abs_x, abs_y]
                        added_keys.append(key)
            
            if added_keys:
                rospy.loginfo_throttle(10, f"🔧 Added fallback keys: {added_keys}")
                
        except Exception as e:
            rospy.logerr(f"Fallback layout error: {e}")
            
        return complete_keypoints

    def calculate_3d_positions_enhanced(self, keypoints_2d):
        """Enhanced 3D position calculation with depth filtering"""
        keypoints_3d = {}
        
        if self.camera_info is None:
            return keypoints_3d

        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4] 
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        for key, (x, y) in keypoints_2d.items():
            try:
                if 0 <= x < self.depth_image.shape[1] and 0 <= y < self.depth_image.shape[0]:
                    # Multi-point depth sampling for robustness
                    depth_samples = []
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            px, py = x + dx, y + dy
                            if (0 <= px < self.depth_image.shape[1] and 
                                0 <= py < self.depth_image.shape[0]):
                                depth_val = self.depth_image[py, px] / 1000.0
                                if 0.1 < depth_val < 2.0:
                                    depth_samples.append(depth_val)
                    
                    if depth_samples:
                        # Use median depth for robustness
                        depth = np.median(depth_samples)
                        
                        # Convert to 3D
                        X = (x - cx) * depth / fx
                        Y = (y - cy) * depth / fy
                        Z = depth
                        
                        keypoints_3d[key] = [X, Y, Z]
                        
            except Exception as e:
                rospy.logdebug(f"3D position error for {key}: {e}")

        return keypoints_3d

    def validate_workspace(self, position):
        """Enhanced workspace validation"""
        try:
            x, y, z = position
            
            # More generous workspace bounds
            workspace_bounds = {
                'x_min': 0.15, 'x_max': 0.85,   # Expanded range
                'y_min': -0.5, 'y_max': 0.5,   # Expanded range
                'z_min': 0.05, 'z_max': 0.6    # Expanded range
            }
            
            in_bounds = (
                workspace_bounds['x_min'] <= x <= workspace_bounds['x_max'] and
                workspace_bounds['y_min'] <= y <= workspace_bounds['y_max'] and
                workspace_bounds['z_min'] <= z <= workspace_bounds['z_max']
            )
            
            return in_bounds
            
        except Exception as e:
            rospy.logerr(f"Workspace validation error: {e}")
            return False

    def detect_keyboard_keys(self, image):
        """Main YOLO key detection method"""
        keypoints_2d = {}
        keyboard_bbox = []
        
        try:
            if self.use_yolo and self.model is not None:
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
                        
                        detected_keys_list = []
                        
                        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                            x1, y1, x2, y2 = box
                            confidence = float(conf)
                            class_id = int(cls)
                            
                            if class_id < len(self.model.names):
                                class_name = self.model.names[class_id]
                            else:
                                continue
                            
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            key_name = self.normalize_key_name(class_name)
                            
                            if key_name and confidence > self.detection_confidence:
                                keypoints_2d[key_name] = [center_x, center_y]
                                detected_keys_list.append(f"{key_name}({confidence:.2f})")
                        
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
                            
        except Exception as e:
            rospy.logerr(f"YOLO detection error: {e}")

        return keypoints_2d, keyboard_bbox

    def normalize_key_name(self, detected_class):
        """Enhanced key name normalization"""
        key_upper = detected_class.upper().strip()
        
        key_mappings = {
            # Letters
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G',
            'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 
            'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
            'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
            
            # Numbers
            '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
            '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
            
            # Special keys
            'SPACE': 'SPACE', 'SPACEBAR': 'SPACE', 'SPACE_BAR': 'SPACE',
            'ENTER': 'ENTER', 'RETURN': 'ENTER',
            'BACKSPACE': 'BACKSPACE', 'DELETE': 'BACKSPACE', 'DEL': 'BACKSPACE',
            'SHIFT': 'SHIFT', 'SHIFT_L': 'SHIFT', 'SHIFT_R': 'SHIFT',
            'CTRL': 'CTRL', 'CONTROL': 'CTRL',
            'ALT': 'ALT', 'TAB': 'TAB', 'ESC': 'ESC', 'ESCAPE': 'ESC'
        }
        
        if key_upper in key_mappings:
            return key_mappings[key_upper]
        
        if len(key_upper) == 1 and (key_upper.isalpha() or key_upper.isdigit()):
            return key_upper
        
        return key_upper

    def transform_to_base_frame(self, point_camera):
        """Enhanced transform with better error handling"""
        try:
            camera_point = PointStamped()
            camera_point.header.frame_id = self.camera_frame
            camera_point.header.stamp = rospy.Time(0)
            camera_point.point.x = point_camera[0]
            camera_point.point.y = point_camera[1]  
            camera_point.point.z = point_camera[2]
            
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, 
                rospy.Time(0), rospy.Duration(3.0)
            )
            
            base_point = do_transform_point(camera_point, transform)
            
            return np.array([base_point.point.x, base_point.point.y, base_point.point.z])
            
        except Exception as e:
            rospy.logerr_throttle(5, f"Transform error: {e}")
            return None

    def publish_annotated_image(self, keypoints_2d, keyboard_bbox):
        """Enhanced visualization"""
        if self.color_image is None:
            return
            
        try:
            annotated = self.color_image.copy()
            
            if keyboard_bbox:
                pts = np.array(keyboard_bbox, np.int32)
                cv2.polylines(annotated, [pts], True, (255, 0, 0), 3)
            
            for key, (x, y) in keypoints_2d.items():
                color = (0, 255, 0) if key in self.keypoints_3d else (0, 165, 255)  # Orange for 2D only
                
                cv2.circle(annotated, (int(x), int(y)), 8, color, -1)
                cv2.circle(annotated, (int(x), int(y)), 10, (255, 255, 255), 2)
                
                font_scale = 0.5
                thickness = 2
                text_size = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                cv2.rectangle(annotated, 
                             (int(x) - text_size[0]//2 - 2, int(y) - text_size[1] - 12),
                             (int(x) + text_size[0]//2 + 2, int(y) - 2),
                             (0, 0, 0), -1)
                
                cv2.putText(annotated, key, 
                           (int(x) - text_size[0]//2, int(y) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            status_lines = [
                f"Model: {'Active' if self.use_yolo else 'Inactive'}",
                f"2D Keys: {len(keypoints_2d)} | 3D Keys: {len(self.keypoints_3d)}",
                f"Status: {'Ready' if self.keyboard_detected else 'Detecting'}",
                f"Typing: {'Active' if self.typing_active else 'Standby'}"
            ]
            
            for i, line in enumerate(status_lines):
                y_pos = 25 + i * 20
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated, (5, y_pos - 15), (text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(annotated, line, (8, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            self.annotated_image_pub.publish(msg)
            
        except Exception as e:
            rospy.logerr(f"Debug image error: {e}")

    def type_text_callback(self, msg):
        if self.typing_active:
            rospy.logwarn("Typing in progress, ignoring request")
            return

        text = msg.data.strip()
        if not text:
            return

        rospy.loginfo(f"📝 Typing request: '{text}'")
        
        typing_thread = threading.Thread(target=self.execute_typing, args=(text,))
        typing_thread.daemon = True
        typing_thread.start()

    def execute_typing(self, text):
        """Enhanced typing execution with smart movement"""
        self.typing_active = True
        self.publish_status(f"Starting to type: '{text}'")

        try:
            if not self.keypoints_3d:
                self.publish_status("❌ No keyboard detected")
                return

            success_count = 0
            total_chars = len(text)
            
            rospy.loginfo(f"🎯 Starting typing sequence for {total_chars} characters")
            
            for i, char in enumerate(text):
                if rospy.is_shutdown():
                    break
                    
                self.publish_status(f"Typing {i+1}/{total_chars}: '{char}'")
                
                key_name = self.char_to_key(char)
                
                if key_name and key_name in self.keypoints_3d:
                    if self.type_key_smart(key_name):
                        success_count += 1
                        rospy.loginfo(f"✅ Typed '{char}' ({i+1}/{total_chars})")
                    else:
                        rospy.logwarn(f"❌ Failed to type '{char}'")
                else:
                    rospy.logwarn(f"⚠️ Key '{key_name}' not available for '{char}'")
                
                # Brief pause between keystrokes
                rospy.sleep(0.3)

            self.publish_status(f"✅ Typing complete: {success_count}/{total_chars}")

        except Exception as e:
            rospy.logerr(f"Typing execution error: {e}")
            self.publish_status(f"❌ Typing error: {str(e)}")
        finally:
            self.typing_active = False

    def char_to_key(self, char):
        """Convert character to key name"""
        if char == ' ':
            return 'SPACE'
        elif char == '\n':
            return 'ENTER'
        elif char == '\b':
            return 'BACKSPACE'
        elif char.isalnum():
            return char.upper()
        else:
            return char.upper()

    def type_key_smart(self, key_name):
        """Smart key typing with optimized movement strategy"""
        if key_name not in self.keypoints_3d:
            return False

        try:
            target_pos = self.keypoints_3d[key_name]['position']
            rospy.loginfo(f"⌨️ Smart typing key '{key_name}' at {target_pos}")

            # Try different strategies in order of preference
            strategies = [
                ("optimized_cartesian", self.type_key_optimized_cartesian),
                ("direct_movement", self.type_key_direct_movement), 
                ("safe_joint_space", self.type_key_safe_joint_space)
            ]
            
            for strategy_name, strategy_func in strategies:
                try:
                    rospy.loginfo(f"🎯 Trying {strategy_name} for '{key_name}'")
                    if strategy_func(key_name, target_pos):
                        rospy.loginfo(f"✅ {strategy_name} successful for '{key_name}'")
                        return True
                    else:
                        rospy.logwarn(f"❌ {strategy_name} failed for '{key_name}'")
                except Exception as e:
                    rospy.logerr(f"❌ {strategy_name} error for '{key_name}': {e}")
                    continue

            rospy.logwarn(f"All typing strategies failed for key '{key_name}'")
            return False

        except Exception as e:
            rospy.logerr(f"Smart key typing error for '{key_name}': {e}")
            return False

    def type_key_optimized_cartesian(self, key_name, target_pos):
        """Optimized Cartesian path with better trajectory planning"""
        try:
            current_pose = self.move_group.get_current_pose().pose
            
            # Create optimized waypoints
            waypoints = []
            
            # Approach point
            approach_pose = copy.deepcopy(current_pose)
            approach_pose.position.x = target_pos[0]
            approach_pose.position.y = target_pos[1]
            approach_pose.position.z = target_pos[2] + self.approach_height
            waypoints.append(approach_pose)
            
            # Press point
            press_pose = copy.deepcopy(approach_pose)
            press_pose.position.z = target_pos[2] - self.press_depth
            waypoints.append(press_pose)
            
            # Retract point
            retract_pose = copy.deepcopy(approach_pose)
            waypoints.append(retract_pose)
            
            # Compute Cartesian path with smaller step size
            rospy.loginfo("Computing optimized Cartesian path...")
            (plan, fraction) = self.move_group.compute_cartesian_path(
                waypoints,
                eef_step=0.005,  # Smaller steps for better trajectory
            )
            
            rospy.loginfo(f"Cartesian path: {fraction*100:.1f}% complete")
            
            if fraction > 0.9:  # Require 90% completion
                # Scale trajectory for safety
                scaled_plan = self.scale_trajectory_timing(plan)
                
                rospy.loginfo("Executing optimized Cartesian trajectory...")
                success = self.move_group.execute(scaled_plan, wait=True)
                
                if success:
                    rospy.sleep(self.dwell_time)
                    self.move_group.stop()
                    return True
                else:
                    rospy.logwarn("Cartesian execution failed")
            else:
                rospy.logwarn(f"Cartesian path only {fraction*100:.1f}% valid")
                
        except Exception as e:
            rospy.logerr(f"Optimized Cartesian error: {e}")
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False

    def type_key_direct_movement(self, key_name, target_pos):
        """Direct movement to key position"""
        try:
            current_pose = self.move_group.get_current_pose().pose
            
            # Direct movement to slightly above the key
            target_pose = copy.deepcopy(current_pose)
            target_pose.position.x = target_pos[0]
            target_pose.position.y = target_pos[1]
            target_pose.position.z = target_pos[2] + 0.003  # Just slightly above
            
            self.move_group.set_pose_target(target_pose)
            self.move_group.set_planning_time(10.0)
            
            rospy.loginfo(f"Planning direct movement to '{key_name}'...")
            plan = self.move_group.plan()
            
            if plan[0]:
                # Scale the plan for safety
                scaled_plan = self.scale_trajectory_timing(plan[1])
                
                success = self.move_group.execute(scaled_plan, wait=True)
                if success:
                    rospy.loginfo(f"✅ Direct movement successful for '{key_name}'")
                    rospy.sleep(self.dwell_time)
                    self.move_group.stop()
                    return True
            else:
                rospy.logwarn(f"Direct movement planning failed for '{key_name}'")
            
        except Exception as e:
            rospy.logerr(f"Direct movement error: {e}")
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False

    def type_key_safe_joint_space(self, key_name, target_pos):
        """Safe joint space movement with multiple waypoints"""
        try:
            current_pose = self.move_group.get_current_pose().pose
            
            # Approach position
            approach_pose = copy.deepcopy(current_pose)
            approach_pose.position.x = target_pos[0]
            approach_pose.position.y = target_pos[1]
            approach_pose.position.z = target_pos[2] + self.approach_height
            
            self.move_group.set_pose_target(approach_pose)
            self.move_group.set_planning_time(15.0)
            
            rospy.loginfo(f"Planning safe approach to '{key_name}'...")
            approach_plan = self.move_group.plan()
            
            if approach_plan[0]:
                scaled_approach = self.scale_trajectory_timing(approach_plan[1])
                
                if self.move_group.execute(scaled_approach, wait=True):
                    rospy.loginfo("✅ Reached safe approach position")
                    
                    # Press movement
                    press_pose = copy.deepcopy(approach_pose)
                    press_pose.position.z = target_pos[2] - self.press_depth
                    
                    self.move_group.set_pose_target(press_pose)
                    press_plan = self.move_group.plan()
                    
                    if press_plan[0]:
                        scaled_press = self.scale_trajectory_timing(press_plan[1])
                        
                        if self.move_group.execute(scaled_press, wait=True):
                            rospy.loginfo(f"✅ Pressed key '{key_name}'")
                            rospy.sleep(self.dwell_time)
                            
                            # Retract
                            self.move_group.set_pose_target(approach_pose)
                            retract_plan = self.move_group.plan()
                            
                            if retract_plan[0]:
                                scaled_retract = self.scale_trajectory_timing(retract_plan[1])
                                self.move_group.execute(scaled_retract, wait=True)
                                rospy.loginfo("✅ Retracted safely")
                            
                            return True
                    
        except Exception as e:
            rospy.logerr(f"Safe joint space error: {e}")
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False

    def scale_trajectory_timing(self, plan):
        """Scale trajectory timing to avoid acceleration limits"""
        try:
            scaled_plan = copy.deepcopy(plan)
            
            if hasattr(scaled_plan, 'joint_trajectory') and scaled_plan.joint_trajectory.points:
                points = scaled_plan.joint_trajectory.points
                
                # Scale time stamps
                time_scale = 2.0  # Make everything 2x slower
                
                for i, point in enumerate(points):
                    point.time_from_start = rospy.Duration(point.time_from_start.to_sec() * time_scale)
                    
                    # Scale velocities and accelerations
                    if point.velocities:
                        point.velocities = [v / time_scale for v in point.velocities]
                    if point.accelerations:
                        point.accelerations = [a / (time_scale * time_scale) for a in point.accelerations]
                
                rospy.loginfo(f"Scaled trajectory timing by factor {time_scale}")
                
            return scaled_plan
            
        except Exception as e:
            rospy.logerr(f"Trajectory scaling error: {e}")
            return plan  # Return original if scaling fails

    def move_to_home_position(self):
        """Move robot to stored home position"""
        try:
            if self.home_position is None:
                rospy.logwarn("No home position stored")
                return False
                
            rospy.loginfo("🏠 Moving to home position...")
            self.move_group.set_pose_target(self.home_position)
            plan = self.move_group.plan()
            
            if plan[0]:
                scaled_plan = self.scale_trajectory_timing(plan[1])
                return self.move_group.execute(scaled_plan, wait=True)
            else:
                rospy.logwarn("Home movement planning failed")
                return False
                
        except Exception as e:
            rospy.logerr(f"Home movement error: {e}")
            return False

    def emergency_stop(self):
        """Emergency stop with enhanced safety"""
        rospy.logwarn("🛑 Emergency stop activated!")
        try:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.typing_active = False
            self.publish_status("🛑 Emergency stop - All motion halted")
            
            # Try to move to a safe position
            rospy.sleep(1.0)  # Brief pause
            self.move_to_home_position()
            
        except Exception as e:
            rospy.logerr(f"Emergency stop error: {e}")

    def shutdown_handler(self):
        """Enhanced shutdown procedure"""
        rospy.loginfo("🔄 Shutting down Robotic Typing Controller...")
        
        try:
            self.typing_active = False
            
            # Stop all movement
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            # Return to home if possible
            self.move_to_home_position()
            
            # Shutdown MoveIt
            moveit_commander.roscpp_shutdown()
            
            rospy.loginfo("✅ Shutdown complete")
            
        except Exception as e:
            rospy.logerr(f"Shutdown error: {e}")

    def get_robot_status(self):
        """Get comprehensive robot status"""
        try:
            status = {
                'timestamp': rospy.Time.now().to_sec(),
                'keyboard_detected': self.keyboard_detected,
                'num_keys_detected': len(self.keypoints_3d),
                'typing_active': self.typing_active,
                'available_keys': list(self.keypoints_3d.keys()) if self.keypoints_3d else [],
                'movement_parameters': {
                    'max_velocity_scale': self.max_velocity_scale,
                    'max_acceleration_scale': self.max_acceleration_scale,
                    'press_depth': self.press_depth,
                    'approach_height': self.approach_height
                }
            }
            
            try:
                current_pose = self.move_group.get_current_pose().pose
                status['current_pose'] = {
                    'position': [current_pose.position.x, current_pose.position.y, current_pose.position.z],
                    'orientation': [current_pose.orientation.x, current_pose.orientation.y, 
                                  current_pose.orientation.z, current_pose.orientation.w]
                }
            except:
                status['current_pose'] = None
                
            return status
            
        except Exception as e:
            rospy.logerr(f"Robot status error: {e}")
            return {}


def main():
    try:
        controller = RoboticTypingController()
        
        rospy.on_shutdown(controller.shutdown_handler)
        
        rospy.loginfo("🤖 Enhanced Robotic Typing Controller is ready!")
        rospy.loginfo("📝 Send text to '/type_text' topic to start typing")
        rospy.loginfo("🔧 Features: Multi-scale detection, Smart movement, Acceleration limiting")
        
        rospy.spin()
        
    except KeyboardInterrupt:
        rospy.loginfo("👋 User interrupted")
    except Exception as e:
        rospy.logfatal(f"❌ Fatal error: {e}")
    finally:
        rospy.loginfo("🔄 Exiting...")

if __name__ == "__main__":
    main()