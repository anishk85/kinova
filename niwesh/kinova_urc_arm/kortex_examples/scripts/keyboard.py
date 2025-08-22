#!/usr/bin/env python3

import rospy
import logging
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
from datetime import datetime


class RoboticTypingController:
    
    def __init__(self):
        rospy.init_node("robotic_typing_controller", log_level=rospy.INFO)
        logging.basicConfig(level=logging.INFO)

        rospy.loginfo("🤖 Initializing Robotic Typing Controller...")

        self.robot_namespace = "my_gen3"  
        self.base_frame = 'base_link'
        self.camera_frame = 'camera_link'

        self.depth_topic = '/camera/depth/image_rect_raw'
        self.color_topic = '/camera/color/image_raw'
        self.camera_info_topic = '/camera/color/camera_info'

        # More conservative movement parameters to avoid acceleration limits
        self.press_depth = 0.015  # Reduced from 0.03 to 1.5cm
        self.dwell_time = 0.3
        self.approach_height = 0.02  # Increased to 2cm above key
        self.safe_height = 0.005

        self.bridge = CvBridge()
        
        # In-memory coordinate storage (no file saving)
        self.keypoints_3d = {}  # Current detected coordinates
        self.session_coordinates = {}  # Accumulated coordinates during session
        
        self.typing_active = False
        self.keyboard_detected = False
        self.depth_image = None
        self.color_image = None
        self.camera_info = None
        self.current_joint_state = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Wait for TF and auto-detect camera frame
        rospy.sleep(1.0)
        self.camera_frame = self.detect_camera_frame()
        rospy.loginfo(f"Using camera frame: {self.camera_frame}")

        self.setup_vision()
        self.load_keyboard_layout()
        self.setup_moveit()
        self.setup_subscribers_publishers()

        rospy.loginfo("✅ Robotic Typing Controller initialized successfully!")

    def detect_camera_frame(self):
        """Automatically detect the correct camera frame for transforms."""
        possible_frames = [
            f'{self.robot_namespace}/camera_color_optical_frame',
            f'{self.robot_namespace}/camera_link',
            f'{self.robot_namespace}/camera_depth_optical_frame',
            'camera_color_optical_frame',
            'camera_link',
            'camera_depth_optical_frame',
            'camera_color_frame',
            'camera_depth_frame'
        ]
        for frame in possible_frames:
            try:
                self.tf_buffer.lookup_transform(self.base_frame, frame, rospy.Time(0), rospy.Duration(1.0))
                rospy.loginfo(f"✅ Found working camera frame: {frame}")
                return frame
            except Exception:
                continue
        rospy.logwarn("❌ No working camera frame found, using 'camera_link' as fallback")
        return 'camera_link'

    def setup_vision(self):
        """Setup YOLO key detection model"""
        try:
            rospy.loginfo("🔧 Setting up YOLO key detection...")
            
            # Use the new best.pt model for key detection
            import os
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
                import logging
                logging.getLogger("ultralytics").setLevel(logging.ERROR)
                
                rospy.loginfo("✅ YOLO key detection model loaded successfully")
            
            # Initialize detection state
            self.last_detection_time = rospy.Time.now()
            self.detection_timeout = 5.0  # seconds
            
        except Exception as e:
            rospy.logerr(f"Vision setup error: {e}")
            self.model = None
            self.use_yolo = False

    def load_keyboard_layout(self):
        """Initialize key detection - no layout file needed"""
        try:
            rospy.loginfo("🔧 Initializing key detection system...")
            self.detected_keys = {}
            rospy.loginfo("✅ Key detection system ready")
        except Exception as e:
            rospy.logerr(f"Error initializing key detection: {e}")

    def setup_moveit(self):
        """Initialize MoveIt components with better trajectory parameters"""
        try:
            rospy.loginfo("🔧 Setting up MoveIt...")
            
            # Initialize MoveIt commander
            moveit_commander.roscpp_initialize([])
            
            # Robot and scene
            self.robot = moveit_commander.RobotCommander(
                robot_description=f"{self.robot_namespace}/robot_description",
                ns=self.robot_namespace
            )
            self.scene = moveit_commander.PlanningSceneInterface(ns=self.robot_namespace)

            # Get available groups
            available_groups = self.robot.get_group_names()
            rospy.loginfo(f"Available planning groups: {available_groups}")
            
            # Use 'arm' group (common for Kinova)
            self.move_group = moveit_commander.MoveGroupCommander(
                "arm",
                robot_description=f"{self.robot_namespace}/robot_description",
                ns=self.robot_namespace
            )
            
            # Configure MoveIt for precise typing with conservative parameters
            self.configure_moveit_for_typing()

            # Much more conservative velocity and acceleration for typing
            self.move_group.set_max_velocity_scaling_factor(0.05)  # 5% max speed (was 10%)
            self.move_group.set_max_acceleration_scaling_factor(0.05)  # 5% max acceleration (was 10%)
            
            rospy.loginfo("✅ MoveIt setup complete.")
            rospy.loginfo(f"   Planning Group: arm")
            rospy.loginfo(f"   Planning Frame: {self.move_group.get_planning_frame()}")
            rospy.loginfo(f"   End Effector: {self.move_group.get_end_effector_link()}")

        except Exception as e:
            rospy.logfatal(f"❌ MoveIt setup failed: {e}")
            raise

    def configure_moveit_for_typing(self):
        """Configure MoveIt parameters for precise typing with conservative settings"""
        self.move_group.set_planning_time(15.0)  # Increased planning time
        self.move_group.set_num_planning_attempts(15)  # More planning attempts
        self.move_group.allow_replanning(True)
        
        self.move_group.set_goal_position_tolerance(0.005)  # Relaxed tolerance (was 0.002)
        self.move_group.set_goal_orientation_tolerance(0.2)  # Relaxed orientation tolerance
        
        self.move_group.set_planner_id("RRTConnect")

    def setup_subscribers_publishers(self):
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        rospy.Subscriber(self.color_topic, Image, self.color_callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(f"/{self.robot_namespace}/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/type_text", String, self.type_text_callback)
        
        self.typing_status_pub = rospy.Publisher("/typing_status", String, queue_size=10)
        self.key_coordinates_pub = rospy.Publisher("/key_coordinates_3d", String, queue_size=10)
        self.annotated_image_pub = rospy.Publisher("/annotated_keyboard", Image, queue_size=10)

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
        """Process key detection and transform to 3D coordinates"""
        if self.color_image is None or self.depth_image is None or self.camera_info is None:
            return

        try:
            # Run YOLO detection on individual keys
            keypoints_2d, keyboard_bbox = self.detect_keyboard_keys(self.color_image)
            
            # Always publish annotated image for debugging
            self.publish_annotated_image(keypoints_2d, keyboard_bbox)
            
            if not keypoints_2d:
                # Check detection timeout
                time_since_detection = (rospy.Time.now() - self.last_detection_time).to_sec()
                if time_since_detection > self.detection_timeout:
                    self.keyboard_detected = False
                return

            # Calculate 3D positions for detected keys
            keypoints_3d_camera = self.calculate_3d_positions(keypoints_2d)
            
            if not keypoints_3d_camera:
                rospy.logwarn_throttle(5, "No 3D positions calculated - check depth image")
                return

            # Transform to base frame and validate workspace
            keypoints_3d_base = {}
            for key, pos_3d_cam in keypoints_3d_camera.items():
                pos_3d_base = self.transform_to_base_frame(pos_3d_cam)
                if pos_3d_base is not None and self.validate_workspace(pos_3d_base):
                    keypoints_3d_base[key] = {
                        'position': pos_3d_base.tolist(),
                        'depth': pos_3d_cam[2],
                        'pixel_coords': keypoints_2d[key],
                        'detection_time': rospy.Time.now().to_sec(),
                        'source': 'live_detection'
                    }
                    
                    # Store in session coordinates (accumulating during session)
                    self.session_coordinates[key] = keypoints_3d_base[key].copy()
                    
                    rospy.loginfo_throttle(5, 
                        f"🎯 Key '{key}': "
                        f"Pixel({keypoints_2d[key][0]}, {keypoints_2d[key][1]}) -> "
                        f"Camera({pos_3d_cam[0]:.3f}, {pos_3d_cam[1]:.3f}, {pos_3d_cam[2]:.3f}) -> "
                        f"Base({pos_3d_base[0]:.3f}, {pos_3d_base[1]:.3f}, {pos_3d_base[2]:.3f})"
                    )

            if keypoints_3d_base:
                # Store fresh detections
                self.keypoints_3d = keypoints_3d_base
                self.keyboard_detected = True
                
                rospy.loginfo_throttle(10, f"✅ Fresh: {len(keypoints_3d_base)} keys | Session total: {len(self.session_coordinates)} keys")
                
                # Print summary of available keys
                fresh_keys = list(keypoints_3d_base.keys())
                all_session_keys = list(self.session_coordinates.keys())
                rospy.loginfo_throttle(15, f"🆕 Fresh keys: {sorted(fresh_keys)}")
                rospy.loginfo_throttle(15, f"📋 Session keys: {sorted(all_session_keys)}")

        except Exception as e:
            rospy.logerr(f"Key detection processing error: {e}")

    def validate_workspace(self, position):
        """Validate if position is within robot's safe workspace"""
        try:
            x, y, z = position
            
            # Define safe workspace bounds for keyboard typing
            workspace_bounds = {
                'x_min': 0.2, 'x_max': 0.8,   # 20cm to 80cm forward
                'y_min': -0.4, 'y_max': 0.4,  # ±40cm left/right  
                'z_min': 0.1, 'z_max': 0.5    # 10cm to 50cm height
            }
            
            in_bounds = (
                workspace_bounds['x_min'] <= x <= workspace_bounds['x_max'] and
                workspace_bounds['y_min'] <= y <= workspace_bounds['y_max'] and
                workspace_bounds['z_min'] <= z <= workspace_bounds['z_max']
            )
            
            if not in_bounds:
                rospy.logdebug_throttle(10, f"Position {position} outside workspace bounds")
                
            return in_bounds
            
        except Exception as e:
            rospy.logerr(f"Workspace validation error: {e}")
            return False

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

    def calculate_3d_positions(self, keypoints_2d):
        """Enhanced 3D position calculation with depth filtering"""
        keypoints_3d = {}
        
        if self.camera_info is None or self.depth_image is None:
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
                        
                        rospy.logdebug(f"Enhanced depth for {key}: {len(depth_samples)} samples, median: {depth:.3f}m")

            except Exception as e:
                rospy.logdebug(f"3D position error for {key}: {e}")

        return keypoints_3d

    def transform_to_base_frame(self, point_camera):
        try:
            camera_point = PointStamped()
            camera_point.header.frame_id = self.camera_frame
            camera_point.header.stamp = rospy.Time(0)
            camera_point.point.x = point_camera[0]
            camera_point.point.y = point_camera[1]
            camera_point.point.z = point_camera[2]

            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(2.0)
            )

            base_point = do_transform_point(camera_point, transform)
            result = np.array([base_point.point.x, base_point.point.y, base_point.point.z])

            return result

        except Exception as e:
            rospy.logerr_throttle(5, f"Transform error ({self.camera_frame} -> {self.base_frame}): {e}")
            return None

    def publish_annotated_image(self, keypoints_2d, keyboard_bbox):
        """Enhanced detection visualization"""
        if self.color_image is None:
            return
            
        try:
            annotated = self.color_image.copy()
            
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
                
                if key in self.keypoints_3d:
                    # Fresh detection
                    depth_data = self.keypoints_3d[key]
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
                elif key in self.session_coordinates:
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
                f"YOLO Model: {'Active' if self.use_yolo else 'Inactive'}",
                f"2D Keys Detected: {len(keypoints_2d)}",
                f"3D Keys (Fresh): {len(self.keypoints_3d)}",
                f"Session Total: {len(self.session_coordinates)}",
                f"Transform Frame: {self.camera_frame}"
            ]
            
            for i, line in enumerate(status_lines):
                y_pos = 30 + i * 25
                cv2.putText(annotated, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Publish debug image
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
        self.typing_active = True
        self.publish_status(f"Starting to type: '{text}'")

        try:
            # Use all available coordinates (fresh + session)
            available_coords = self.session_coordinates.copy()
            available_coords.update(self.keypoints_3d)  # Fresh coords override session coords
            
            if not available_coords:
                self.publish_status("❌ No keyboard coordinates available")
                return

            success_count = 0
            total_chars = len(text)
            
            for i, char in enumerate(text):
                if rospy.is_shutdown():
                    break
                    
                self.publish_status(f"Typing {i+1}/{total_chars}: '{char}'")
                
                key_name = self.char_to_key(char)
                
                if key_name and key_name in available_coords:
                    source = "fresh detection" if key_name in self.keypoints_3d else "session coordinates"
                    rospy.loginfo(f"🎯 Using {source} for key '{key_name}'")
                    
                    if self.type_key(key_name, available_coords):
                        success_count += 1
                        rospy.loginfo(f"✅ Typed '{char}' using {source}")
                    else:
                        rospy.logwarn(f"❌ Failed to type '{char}'")
                else:
                    rospy.logwarn(f"⚠️ Key '{key_name}' not available for '{char}'")
                    rospy.loginfo(f"📋 Available keys: {list(available_coords.keys())}")
                
                rospy.sleep(0.8)  # Longer delay between keystrokes

            self.publish_status(f"✅ Typing complete: {success_count}/{total_chars}")

        except Exception as e:
            rospy.logerr(f"Typing execution error: {e}")
            self.publish_status(f"❌ Typing error: {str(e)}")
        finally:
            self.typing_active = False

    def char_to_key(self, char):
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

    def type_key(self, key_name, available_coords):
        if key_name not in available_coords:
            return False

        try:
            target_pos = available_coords[key_name]['position']
            rospy.loginfo(f"⌨️ Typing key '{key_name}' at {target_pos}")

            # Try improved Cartesian approach first
            if self.type_key_improved_cartesian(key_name, target_pos):
                return True

            # Fallback to joint space approach
            if self.type_key_joint_space(key_name, target_pos):
                return True

            rospy.logwarn(f"All typing methods failed for key '{key_name}'")
            return False

        except Exception as e:
            rospy.logerr(f"Key typing error for '{key_name}': {e}")
            return False

    def type_key_improved_cartesian(self, key_name, target_pos):
        """
        MODIFIED Cartesian approach with a horizontal press motion.
        """
        try:
            rospy.loginfo(f"🎯 Horizontal Cartesian approach for '{key_name}'")
            current_pose = self.move_group.get_current_pose().pose

            # --- Define Horizontal Motion Parameters ---
            # How far in front of the key to start the motion (in meters)
            horizontal_offset = 0.05  # 5 cm
            # How high above the key surface to move (in meters)
            vertical_clearance = 0.01 # 1 cm
            x_press_offset = 0.07

            waypoints = []

            # 1. Move to a safe point: IN FRONT of and ABOVE the key
            safe_pose = Pose()
            safe_pose.position.x = target_pos[0] - horizontal_offset # Move back along X-axis
            safe_pose.position.y = target_pos[1]
            safe_pose.position.z = target_pos[2] + vertical_clearance # Move up
            safe_pose.orientation = current_pose.orientation
            waypoints.append(safe_pose)

            # 2. Move to pre-press position: IN FRONT of the key, at the correct height
            pre_press_pose = Pose()
            pre_press_pose.position.x = target_pos[0]  # Still back along X
            pre_press_pose.position.y = target_pos[1]
            pre_press_pose.position.z = target_pos[2] # At target height
            pre_press_pose.orientation = current_pose.orientation
            waypoints.append(pre_press_pose)

            # 3. Press the key: Move FORWARD horizontally
            press_pose = Pose()
            press_pose.position.x = target_pos[0] + x_press_offset # Move to target X
            press_pose.position.y = target_pos[1]
            press_pose.position.z = target_pos[2]
            press_pose.orientation = current_pose.orientation
            waypoints.append(press_pose)

            # 4. Release: Move BACK horizontally to the pre-press position
            waypoints.append(pre_press_pose)
            
            # 5. Retract: Move back to the safe position
            waypoints.append(safe_pose)

            rospy.loginfo("Computing horizontal Cartesian path...")
            (plan, fraction) = self.move_group.compute_cartesian_path(
                waypoints,
                eef_step=0.002,
                 # Disable jump threshold
            )

            rospy.loginfo(f"Cartesian path: {fraction*100:.1f}% complete")

            if fraction > 0.7:
                if hasattr(plan, 'joint_trajectory'):
                    self.scale_trajectory_timing(plan.joint_trajectory)
                
                rospy.loginfo("Executing horizontal Cartesian trajectory...")
                success = self.move_group.execute(plan, wait=True)
                
                if success:
                    rospy.loginfo(f"✅ Horizontal Cartesian typing successful for '{key_name}'")
                    # Dwell time is effectively handled by the motion itself, but a small sleep can help
                    rospy.sleep(0.1) 
                    self.move_group.stop()
                    return True
                else:
                    rospy.logwarn("Horizontal Cartesian execution failed")
            else:
                rospy.logwarn(f"Horizontal Cartesian path only {fraction*100:.1f}% valid, trying step-by-step")
                if self.execute_waypoints_stepwise(waypoints):
                    return True

        except Exception as e:
            rospy.logerr(f"Horizontal Cartesian typing error: {e}")
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()

        return False
    def scale_trajectory_timing(self, joint_trajectory):
        """Scale trajectory timing to reduce velocities and accelerations"""
        try:
            # Scale time stamps to make movement slower
            time_scale_factor = 3.0  # Make 3x slower
            
            for i, point in enumerate(joint_trajectory.points):
                if i == 0:
                    continue  # Skip first point
                
                # Scale time
                point.time_from_start = rospy.Duration(
                    point.time_from_start.to_sec() * time_scale_factor
                )
                
                # Reduce velocities
                if point.velocities:
                    point.velocities = [v / time_scale_factor for v in point.velocities]
                
                # Reduce accelerations  
                if point.accelerations:
                    point.accelerations = [a / (time_scale_factor * time_scale_factor) for a in point.accelerations]
                    
        except Exception as e:
            rospy.logerr(f"Trajectory scaling error: {e}")

    def execute_waypoints_stepwise(self, waypoints):
        """Execute waypoints one by one instead of as a single trajectory"""
        try:
            rospy.loginfo("Executing waypoints step-by-step...")
            
            for i, waypoint in enumerate(waypoints):
                rospy.loginfo(f"Moving to waypoint {i+1}/{len(waypoints)}")
                
                self.move_group.set_pose_target(waypoint)
                self.move_group.set_planning_time(10.0)
                
                plan = self.move_group.plan()
                
                if plan[0]:  # If planning succeeded
                    success = self.move_group.execute(plan[1], wait=True)
                    if success:
                        rospy.loginfo(f"✅ Reached waypoint {i+1}")
                        
                        # Add dwell time for press waypoint
                        if i == 2:  # Press waypoint (0-indexed)
                            rospy.sleep(self.dwell_time)
                    else:
                        rospy.logwarn(f"❌ Failed to reach waypoint {i+1}")
                        return False
                else:
                    rospy.logwarn(f"❌ Planning failed for waypoint {i+1}")
                    return False
                
                rospy.sleep(0.2)  # Small delay between waypoints
            
            rospy.loginfo("✅ Step-by-step execution completed")
            return True
            
        except Exception as e:
            rospy.logerr(f"Step-by-step execution error: {e}")
            return False
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()

    def type_key_joint_space(self, key_name, target_pos):
        """Joint space typing with better error handling and faster movement"""
        try:
            rospy.loginfo(f"🔧 Joint space approach for '{key_name}'")
            
            current_pose = self.move_group.get_current_pose().pose
            
            # Temporarily increase velocity for joint space movements
            original_vel_scaling = self.move_group.get_max_velocity_scaling_factor()
            original_acc_scaling = self.move_group.get_max_acceleration_scaling_factor()
            
            # Set faster movement for joint space (but still conservative)
            self.move_group.set_max_velocity_scaling_factor(0.15)  # Increased from 0.05 to 0.15
            self.move_group.set_max_acceleration_scaling_factor(0.15)  # Increased from 0.05 to 0.15
            
            # Approach position
            approach_pose = Pose()
            approach_pose.position.x = target_pos[0]
            approach_pose.position.y = target_pos[1]
            approach_pose.position.z = target_pos[2] + self.approach_height
            approach_pose.orientation = current_pose.orientation
            
            self.move_group.set_pose_target(approach_pose)
            self.move_group.set_planning_time(10.0)  # Reduced from 15.0
            
            rospy.loginfo("Planning approach movement...")
            approach_plan = self.move_group.plan()
            
            if approach_plan[0]: 
                rospy.loginfo("Executing approach...")
                if self.move_group.execute(approach_plan[1], wait=True):
                    rospy.loginfo("✅ Reached approach position")
                    
                    # Press position - use slower movement for precision
                    self.move_group.set_max_velocity_scaling_factor(0.15)  # Slower for press
                    self.move_group.set_max_acceleration_scaling_factor(0.15)
                    
                    press_pose = Pose()
                    press_pose.position.x = target_pos[0]
                    press_pose.position.y = target_pos[1]
                    press_pose.position.z = target_pos[2] - self.press_depth
                    press_pose.orientation = current_pose.orientation
                    
                    self.move_group.set_pose_target(press_pose)
                    press_plan = self.move_group.plan()
                    
                    if press_plan[0]:
                        rospy.loginfo("Executing key press...")
                        if self.move_group.execute(press_plan[1], wait=True):
                            rospy.loginfo(f"✅ Pressed key '{key_name}'")
                            rospy.sleep(self.dwell_time)
                            
                            # Retract - faster movement again
                            self.move_group.set_max_velocity_scaling_factor(0.05)
                            self.move_group.set_max_acceleration_scaling_factor(0.05)
                            
                            self.move_group.set_pose_target(approach_pose)
                            retract_plan = self.move_group.plan()
                            
                            if retract_plan[0]:
                                self.move_group.execute(retract_plan[1], wait=True)
                                rospy.loginfo("✅ Retracted from key")
                            
                            return True
                    else:
                        rospy.logwarn("Press movement planning failed")
                else:
                    rospy.logwarn("Approach movement execution failed")
            else:
                rospy.logwarn("Approach movement planning failed")
                
        except Exception as e:
            rospy.logerr(f"Joint space typing error: {e}")
            
        finally:
            # Restore original velocity scaling
            try:
                self.move_group.set_max_velocity_scaling_factor(original_vel_scaling)
                self.move_group.set_max_acceleration_scaling_factor(original_acc_scaling)
            except:
                # Fallback to default conservative values
                self.move_group.set_max_velocity_scaling_factor(0.05)
                self.move_group.set_max_acceleration_scaling_factor(0.05)
                
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False


    def get_robot_status(self):
        try:
            status = {
                'timestamp': rospy.Time.now().to_sec(),
                'keyboard_detected': self.keyboard_detected,
                'num_keys_detected': len(self.keypoints_3d),
                'num_session_keys': len(self.session_coordinates),
                'typing_active': self.typing_active,
                'current_pose': None,
                'joint_states': None
            }
            
            try:
                current_pose = self.move_group.get_current_pose().pose
                status['current_pose'] = {
                    'position': [current_pose.position.x, current_pose.position.y, current_pose.position.z],
                    'orientation': [current_pose.orientation.x, current_pose.orientation.y, 
                                  current_pose.orientation.z, current_pose.orientation.w]
                }
            except:
                pass
                
            if self.current_joint_state:
                status['joint_states'] = {
                    'names': list(self.current_joint_state.name),
                    'positions': list(self.current_joint_state.position)
                }
                
            return status
            
        except Exception as e:
            rospy.logerr(f"Robot status error: {e}")
            return {}

    def emergency_stop(self):
        rospy.logwarn("🛑 Emergency stop!")
        try:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.typing_active = False
            self.publish_status("🛑 Emergency stop activated")
            rospy.logwarn("All robot motion stopped")
        except Exception as e:
            rospy.logerr(f"Emergency stop error: {e}")

    def shutdown_handler(self):
        rospy.loginfo("🔄 Shutting down Robotic Typing Controller...")
        
        try:
            self.typing_active = False
            
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
            moveit_commander.roscpp_shutdown()
            
            rospy.loginfo(f"✅ Session ended with {len(self.session_coordinates)} keys learned")
            rospy.loginfo("✅ Shutdown complete")
            
        except Exception as e:
            rospy.logerr(f"Shutdown error: {e}")


def main():
    try:
        controller = RoboticTypingController()
        
        rospy.on_shutdown(controller.shutdown_handler)
        
        rospy.loginfo("🤖 Robotic Typing Controller is ready!")
        rospy.loginfo("📝 Send text to '/type_text' topic to start typing")
        rospy.loginfo("🎨 Watch '/annotated_keyboard' topic for visual feedback")
        rospy.loginfo("   - GREEN keys: Fresh detection with good depth")
        rospy.loginfo("   - YELLOW keys: Session coordinates available")
        rospy.loginfo("   - RED keys: Poor or no depth data")
        rospy.loginfo("💾 Coordinates are stored in memory during this session only")
        
        rospy.spin()
        
    except KeyboardInterrupt:
        rospy.loginfo("👋 User interrupted")
    except Exception as e:
        rospy.logfatal(f"❌ Fatal error: {e}")
    finally:
        rospy.loginfo("🔄 Exiting...")


if __name__ == "__main__":
    main()