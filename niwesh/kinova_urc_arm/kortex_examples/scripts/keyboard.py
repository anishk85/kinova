#!/usr/bin/env python3

import rospy
import logging
import os
import json
import numpy as np
import torch  # Add this import
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
        self.camera_frame = None  # Will auto-detect below

        self.depth_topic = '/camera/depth/image_rect_raw'
        self.color_topic = '/camera/color/image_raw'
        self.camera_info_topic = '/camera/color/camera_info'

        self.press_depth = 0.002  # 2mm press
        self.dwell_time = 0.2
        self.approach_height = 0.01  # 1cm above key
        self.safe_height = 0.05

        self.bridge = CvBridge()
        self.keypoints_3d = {}
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

        # Initialize coordinate saving system BEFORE other setup
        self.setup_coordinate_saving()

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

    def setup_coordinate_saving(self):
        """Initialize coordinate saving system"""
        try:
            # Create coordinates directory if it doesn't exist
            self.coordinates_dir = os.path.expanduser("~/keyboard_coordinates")
            if not os.path.exists(self.coordinates_dir):
                os.makedirs(self.coordinates_dir)
                rospy.loginfo(f"📁 Created coordinates directory: {self.coordinates_dir}")
            
            # File paths
            self.saved_coordinates_file = os.path.join(self.coordinates_dir, "saved_key_coordinates.json")
            self.backup_coordinates_file = os.path.join(self.coordinates_dir, "backup_key_coordinates.json")
            
            # Load existing coordinates if available
            self.saved_keypoints_3d = self.load_saved_coordinates()
            
            # Coordinate saving mode
            self.coordinate_saving_mode = False
            
            rospy.loginfo("✅ Coordinate saving system initialized")
            rospy.loginfo(f"📂 Coordinates will be saved to: {self.saved_coordinates_file}")
            
        except Exception as e:
            rospy.logerr(f"Coordinate saving setup error: {e}")
            self.saved_keypoints_3d = {}

    def load_saved_coordinates(self):
        """Load previously saved key coordinates"""
        try:
            if os.path.exists(self.saved_coordinates_file):
                with open(self.saved_coordinates_file, 'r') as f:
                    data = json.load(f)
                    
                saved_coords = data.get('keypoints_3d', {})
                metadata = data.get('metadata', {})
                
                rospy.loginfo(f"📖 Loaded {len(saved_coords)} saved key coordinates")
                rospy.loginfo(f"💾 Last saved: {metadata.get('timestamp', 'Unknown')}")
                rospy.loginfo(f"🔑 Available saved keys: {list(saved_coords.keys())}")
                
                return saved_coords
            else:
                rospy.loginfo("📝 No saved coordinates found, starting fresh")
                return {}
                
        except Exception as e:
            rospy.logerr(f"Error loading saved coordinates: {e}")
            return {}

    def save_coordinates(self, coordinates_to_save=None):
        """Save current key coordinates with good depth data"""
        try:
            if coordinates_to_save is None:
                coordinates_to_save = self.keypoints_3d.copy()
            
            # Filter only keys with good depth data (marked green in visualization)
            good_depth_keys = {}
            for key, data in coordinates_to_save.items():
                if isinstance(data, dict) and 'position' in data and 'depth' in data:
                    # Check if depth is reasonable (between 0.1m and 2.0m)
                    if 0.1 < data['depth'] < 2.0:
                        good_depth_keys[key] = data
            
            if not good_depth_keys:
                rospy.logwarn("⚠️ No keys with good depth data found to save")
                return False
            
            # Create backup of existing file
            if os.path.exists(self.saved_coordinates_file):
                import shutil
                shutil.copy2(self.saved_coordinates_file, self.backup_coordinates_file)
                rospy.loginfo("💾 Created backup of existing coordinates")
            
            # Merge with existing coordinates
            merged_coordinates = self.saved_keypoints_3d.copy()
            merged_coordinates.update(good_depth_keys)
            
            # Prepare data to save
            save_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'total_keys': len(merged_coordinates),
                    'newly_added': len(good_depth_keys),
                    'camera_frame': self.camera_frame,
                    'base_frame': self.base_frame,
                    'robot_namespace': self.robot_namespace
                },
                'keypoints_3d': merged_coordinates
            }
            
            # Save to file
            with open(self.saved_coordinates_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            # Update internal saved coordinates
            self.saved_keypoints_3d = merged_coordinates
            
            rospy.loginfo("✅ Successfully saved key coordinates!")
            rospy.loginfo(f"💾 Saved {len(good_depth_keys)} keys with good depth data")
            rospy.loginfo(f"🔑 Keys saved: {list(good_depth_keys.keys())}")
            rospy.loginfo(f"📊 Total saved keys: {len(merged_coordinates)}")
            
            return True
            
        except Exception as e:
            rospy.logerr(f"Error saving coordinates: {e}")
            return False

    def merge_coordinates_with_saved(self):
        """Merge current detected coordinates with saved coordinates"""
        try:
            merged_coordinates = {}
            
            # Start with saved coordinates
            merged_coordinates.update(self.saved_keypoints_3d)
            
            # Override with freshly detected coordinates (these are more current)
            for key, data in self.keypoints_3d.items():
                if isinstance(data, dict) and 'depth' in data:
                    # Only use if depth is reasonable
                    if 0.1 < data['depth'] < 2.0:
                        merged_coordinates[key] = data
                        rospy.logdebug(f"🔄 Using fresh coordinates for '{key}'")
                    elif key in self.saved_keypoints_3d:
                        rospy.logdebug(f"💾 Using saved coordinates for '{key}' (poor fresh depth)")
                elif key in self.saved_keypoints_3d:
                    rospy.logdebug(f"💾 Using saved coordinates for '{key}' (no fresh detection)")
            
            # Update keypoints for typing
            original_count = len(self.keypoints_3d)
            self.keypoints_3d = merged_coordinates
            new_count = len(self.keypoints_3d)
            
            if new_count > original_count:
                rospy.loginfo(f"🔄 Merged coordinates: {original_count} -> {new_count} keys available")
                
            return merged_coordinates
            
        except Exception as e:
            rospy.logerr(f"Error merging coordinates: {e}")
            return self.keypoints_3d

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
        """Initialize MoveIt components"""
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
            
            # Configure MoveIt for precise typing
            self.configure_moveit_for_typing()

            # Set velocity and acceleration scaling for safer typing
            self.move_group.set_max_velocity_scaling_factor(0.1)  # 10% max speed
            self.move_group.set_max_acceleration_scaling_factor(0.1)  # 10% max acceleration
            
            rospy.loginfo("✅ MoveIt setup complete.")
            rospy.loginfo(f"   Planning Group: arm")
            rospy.loginfo(f"   Planning Frame: {self.move_group.get_planning_frame()}")
            rospy.loginfo(f"   End Effector: {self.move_group.get_end_effector_link()}")

        except Exception as e:
            rospy.logfatal(f"❌ MoveIt setup failed: {e}")
            raise

    def configure_moveit_for_typing(self):
        """Configure MoveIt parameters for precise typing"""
        self.move_group.set_planning_time(10.0)
        self.move_group.set_num_planning_attempts(10)
        self.move_group.allow_replanning(True)
        
        self.move_group.set_goal_position_tolerance(0.002)  
        self.move_group.set_goal_orientation_tolerance(0.1)  
        
        self.move_group.set_planner_id("RRTConnect")

    def setup_subscribers_publishers(self):
        rospy.Subscriber(self.depth_topic, Image, self.depth_callback)
        rospy.Subscriber(self.color_topic, Image, self.color_callback)
        rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        rospy.Subscriber(f"/{self.robot_namespace}/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/type_text", String, self.type_text_callback)
        
        # Add new subscribers for coordinate saving
        rospy.Subscriber("/save_coordinates", String, self.save_coordinates_callback)
        rospy.Subscriber("/toggle_coordinate_saving", String, self.toggle_coordinate_saving_callback)
        
        self.typing_status_pub = rospy.Publisher("/typing_status", String, queue_size=10)
        self.key_coordinates_pub = rospy.Publisher("/key_coordinates_3d", String, queue_size=10)
        self.annotated_image_pub = rospy.Publisher("/annotated_keyboard", Image, queue_size=10)
        self.trajectory_pub = rospy.Publisher(
            f"/{self.robot_namespace}/move_group/display_planned_path", 
            DisplayTrajectory, queue_size=10
        )
        
        # Add publisher for coordinate saving status
        self.coordinate_status_pub = rospy.Publisher("/coordinate_saving_status", String, queue_size=10)

    def save_coordinates_callback(self, msg):
        """Callback to save current coordinates"""
        rospy.loginfo("💾 Manual coordinate saving requested...")
        success = self.save_coordinates()
        
        status_msg = "✅ Coordinates saved successfully!" if success else "❌ Failed to save coordinates"
        self.coordinate_status_pub.publish(String(data=status_msg))

    def toggle_coordinate_saving_callback(self, msg):
        """Toggle coordinate saving mode"""
        self.coordinate_saving_mode = not self.coordinate_saving_mode
        status = "ON" if self.coordinate_saving_mode else "OFF"
        
        rospy.loginfo(f"🔄 Coordinate saving mode: {status}")
        self.coordinate_status_pub.publish(
            String(data=f"Coordinate saving mode: {status}")
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
                    
                # Still merge with saved coordinates even if no new detection
                if self.saved_keypoints_3d:
                    self.merge_coordinates_with_saved()
                    
                return

            # Calculate 3D positions for detected keys
            keypoints_3d_camera = self.calculate_3d_positions(keypoints_2d)
            
            if not keypoints_3d_camera:
                rospy.logwarn_throttle(5, "No 3D positions calculated - check depth image")
                
                # Still merge with saved coordinates
                if self.saved_keypoints_3d:
                    self.merge_coordinates_with_saved()
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
                    
                    # Print coordinates for debugging
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
                
                # Merge with saved coordinates
                merged_coordinates = self.merge_coordinates_with_saved()
                
                # Auto-save if in coordinate saving mode
                if self.coordinate_saving_mode:
                    rospy.loginfo_throttle(10, "💾 Auto-saving coordinates (saving mode active)")
                    self.save_coordinates(keypoints_3d_base)
                
                # Publish key coordinates with detection info
                detection_data = {
                    "timestamp": rospy.Time.now().to_sec(),
                    "detection_method": "YOLO_individual_keys",
                    "total_keys_detected": len(keypoints_3d_base),
                    "total_keys_available": len(merged_coordinates),
                    "saved_keys_loaded": len(self.saved_keypoints_3d),
                    "keys": merged_coordinates,
                    "fresh_detections": keypoints_3d_base
                }
                
                self.key_coordinates_pub.publish(String(data=json.dumps(detection_data)))
                
                rospy.loginfo_throttle(10, f"✅ Fresh: {len(keypoints_3d_base)} keys | Total available: {len(merged_coordinates)} keys")
                
                # Print summary of available keys
                fresh_keys = list(keypoints_3d_base.keys())
                all_keys = list(merged_coordinates.keys())
                rospy.loginfo_throttle(15, f"🆕 Fresh keys: {sorted(fresh_keys)}")
                rospy.loginfo_throttle(15, f"📋 All available keys: {sorted(all_keys)}")

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
                                
                                rospy.logdebug(f"Detected key '{key_name}' at ({center_x}, {center_y}) confidence: {confidence:.2f}")
                        
                        # Print detected keys periodically
                        if detected_keys_list:
                            rospy.loginfo_throttle(2, f"🔍 Detected keys: {detected_keys_list}")
                        
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
                    # Median filter over 5x5 window
                    window = self.depth_image[max(0, y-2):y+3, max(0, x-2):x+3]
                    depth_vals = window.flatten()
                    depth_vals = depth_vals[(depth_vals > 100) & (depth_vals < 2000)]  # 0.1m to 2.0m in mm
                    if len(depth_vals) == 0:
                        continue
                    depth = np.median(depth_vals) / 1000.0  # meters

                    if 0.1 < depth < 2.0:
                        X = (x - cx) * depth / fx
                        Y = (y - cy) * depth / fy
                        Z = depth
                        keypoints_3d[key] = [X, Y, Z]
            except Exception as e:
                rospy.logwarn_throttle(10, f"3D position error for {key}: {e}")

        return keypoints_3d

    def transform_to_base_frame(self, point_camera):
        try:
            camera_point = PointStamped()
            camera_point.header.frame_id = self.camera_frame
            camera_point.header.stamp = rospy.Time(0)
            camera_point.point.x = point_camera[0]
            camera_point.point.y = point_camera[1]
            camera_point.point.z = point_camera[2]

            rospy.loginfo(f"Transforming point {point_camera} from {self.camera_frame} to {self.base_frame}")

            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame,
                rospy.Time(0), rospy.Duration(2.0)
            )

            base_point = do_transform_point(camera_point, transform)
            result = np.array([base_point.point.x, base_point.point.y, base_point.point.z])

            rospy.loginfo(f"Transformed to base: {result}")

            return result

        except Exception as e:
            rospy.logerr_throttle(5, f"Transform error ({self.camera_frame} -> {self.base_frame}): {e}")
            return None

    def publish_annotated_image(self, keypoints_2d, keyboard_bbox):
        """Enhanced detection visualization with coordinate saving info"""
        if self.color_image is None:
            return
            
        try:
            # Enhanced annotation for debugging
            annotated = self.color_image.copy()
            
            # Draw keyboard bounding box if available
            if keyboard_bbox:
                pts = np.array(keyboard_bbox, np.int32)
                cv2.polylines(annotated, [pts], True, (255, 0, 0), 3)
                cv2.putText(annotated, "Keyboard Region", 
                           (int(pts[0][0]), int(pts[0][1] - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Get merged coordinates for visualization
            merged_coords = self.merge_coordinates_with_saved() if hasattr(self, 'saved_keypoints_3d') else self.keypoints_3d
            
            # Draw detected keys with enhanced visualization
            for key, (x, y) in keypoints_2d.items():
                # Color coding: 
                # Green = fresh detection with good depth
                # Blue = fresh detection but poor depth
                # Yellow = using saved coordinates
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
                elif key in self.saved_keypoints_3d:
                    color = (0, 255, 255)  # Yellow - using saved coordinates
                    status = "SAVED"
                else:
                    color = (0, 0, 255)  # Red - no 3D data
                    status = "NO_3D"
                
                # Draw key center with larger circle
                cv2.circle(annotated, (int(x), int(y)), 10, color, -1)
                cv2.circle(annotated, (int(x), int(y)), 12, (255, 255, 255), 2)
                
                # Draw key label with background and status
                font_scale = 0.5
                thickness = 2
                label = f"{key} ({status})"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Background rectangle for text
                cv2.rectangle(annotated, 
                             (int(x) - text_size[0]//2 - 3, int(y) - text_size[1] - 20),
                             (int(x) + text_size[0]//2 + 3, int(y) - 5),
                             (0, 0, 0), -1)
                
                # Text
                cv2.putText(annotated, label, 
                           (int(x) - text_size[0]//2, int(y) - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
            
            # Enhanced status information
            status_lines = [
                f"YOLO Model: {'Active' if self.use_yolo else 'Inactive'}",
                f"2D Keys Detected: {len(keypoints_2d)}",
                f"3D Keys (Fresh): {len(self.keypoints_3d)}",
                f"3D Keys (Saved): {len(self.saved_keypoints_3d)}",
                f"Total Available: {len(merged_coords)}",
                f"Coordinate Saving: {'ON' if self.coordinate_saving_mode else 'OFF'}",
                f"Transform Frame: {self.camera_frame}"
            ]
            
            for i, line in enumerate(status_lines):
                y_pos = 30 + i * 25
                # Background for text
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated, (5, y_pos - 20), (text_size[0] + 10, y_pos + 5), (0, 0, 0), -1)
                cv2.putText(annotated, line, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add legend for colors
            legend_y_start = annotated.shape[0] - 120
            legend_items = [
                ("GREEN: Fresh + Good Depth", (0, 255, 0)),
                ("YELLOW: Using Saved Coords", (0, 255, 255)),
                ("RED: Poor/No Depth", (0, 0, 255))
            ]
            
            for i, (text, color) in enumerate(legend_items):
                y_pos = legend_y_start + i * 25
                cv2.rectangle(annotated, (10, y_pos - 15), (30, y_pos + 5), color, -1)
                cv2.putText(annotated, text, (40, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
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
            # Use merged coordinates (fresh + saved)
            available_coords = self.merge_coordinates_with_saved()
            
            if not available_coords:
                self.publish_status("❌ No keyboard coordinates available (neither detected nor saved)")
                return

            success_count = 0
            total_chars = len(text)
            
            for i, char in enumerate(text):
                if rospy.is_shutdown():
                    break
                    
                self.publish_status(f"Typing {i+1}/{total_chars}: '{char}'")
                
                key_name = self.char_to_key(char)
                
                if key_name and key_name in available_coords:
                    source = "fresh detection" if key_name in self.keypoints_3d else "saved coordinates"
                    rospy.loginfo(f"🎯 Using {source} for key '{key_name}'")
                    
                    if self.type_key(key_name):
                        success_count += 1
                        rospy.loginfo(f"✅ Typed '{char}' using {source}")
                    else:
                        rospy.logwarn(f"❌ Failed to type '{char}'")
                else:
                    rospy.logwarn(f"⚠️ Key '{key_name}' not available for '{char}'")
                    rospy.loginfo(f"📋 Available keys: {list(available_coords.keys())}")
                
                rospy.sleep(0.5)

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

    def type_key(self, key_name):
        # Use merged coordinates
        available_coords = self.merge_coordinates_with_saved()
        
        if key_name not in available_coords:
            return False

        try:
            target_pos = available_coords[key_name]['position']
            rospy.loginfo(f"⌨️ Typing key '{key_name}' at {target_pos}")

            if self.type_key_cartesian(key_name, target_pos):
                return True

            if self.type_key_joint_space(key_name, target_pos):
                return True

            if self.type_key_simple_pose(key_name, target_pos):
                return True

            rospy.logwarn(f"All typing methods failed for key '{key_name}'")
            return False

        except Exception as e:
            rospy.logerr(f"Key typing error for '{key_name}': {e}")
            return False
            
    def type_key_cartesian(self, key_name, target_pos):
        try:
            rospy.loginfo(f"🎯 Cartesian approach for '{key_name}'")
            current_pose = self.move_group.get_current_pose().pose

            # Log all relevant info
            rospy.loginfo(f"Current EE pose: x={current_pose.position.x:.4f}, y={current_pose.position.y:.4f}, z={current_pose.position.z:.4f}")
            rospy.loginfo(f"Target key '{key_name}' position: {target_pos} (should be in {self.base_frame})")
            rospy.loginfo(f"Planning frame: {self.move_group.get_planning_frame()}")
            rospy.loginfo(f"End effector link: {self.move_group.get_end_effector_link()}")

            waypoints = []

            # Approach above key
            approach_pose = Pose()
            approach_pose.position.x = target_pos[0]
            approach_pose.position.y = target_pos[1]
            approach_pose.position.z = target_pos[2] + self.approach_height
            approach_pose.orientation = current_pose.orientation
            waypoints.append(approach_pose)

            # Press key
            press_pose = Pose()
            press_pose.position.x = target_pos[0]
            press_pose.position.y = target_pos[1]
            press_pose.position.z = target_pos[2] - self.press_depth
            press_pose.orientation = current_pose.orientation
            waypoints.append(press_pose)

            # Retract
            retract_pose = Pose()
            retract_pose.position.x = target_pos[0]
            retract_pose.position.y = target_pos[1]
            retract_pose.position.z = target_pos[2] + self.approach_height
            retract_pose.orientation = current_pose.orientation
            waypoints.append(retract_pose)

            rospy.loginfo("Computing Cartesian path...")
            (plan, fraction) = self.move_group.compute_cartesian_path(
                waypoints,
                eef_step=0.005,  # 5mm steps for more accuracy
            )

            rospy.loginfo(f"Cartesian path: {fraction*100:.1f}% complete")

            if fraction > 0.8:
                rospy.loginfo("Executing Cartesian trajectory...")
                success = self.move_group.execute(plan, wait=True)
                if success:
                    rospy.loginfo(f"✅ Cartesian typing successful for '{key_name}'")
                    rospy.sleep(self.dwell_time)
                    self.move_group.stop()
                    return True
                else:
                    rospy.logwarn("Cartesian execution failed")
            else:
                rospy.logwarn(f"Cartesian path only {fraction*100:.1f}% valid")

        except Exception as e:
            rospy.logerr(f"Cartesian typing error: {e}")

        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()

        return False

    def type_key_joint_space(self, key_name, target_pos):
        try:
            rospy.loginfo(f"🔧 Joint space approach for '{key_name}'")
            
            current_pose = self.move_group.get_current_pose().pose
            
            approach_pose = Pose()
            approach_pose.position.x = target_pos[0]
            approach_pose.position.y = target_pos[1]
            approach_pose.position.z = target_pos[2] + self.approach_height
            approach_pose.orientation = current_pose.orientation
            
            self.move_group.set_pose_target(approach_pose)
            
            rospy.loginfo("Planning approach movement...")
            approach_plan = self.move_group.plan()
            
            if approach_plan[0]: 
                rospy.loginfo("Executing approach...")
                if self.move_group.execute(approach_plan[1], wait=True):
                    rospy.loginfo("✅ Reached approach position")
                    
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
                            
                            self.move_group.set_pose_target(approach_pose)
                            retract_plan = self.move_group.plan()
                            
                            if retract_plan[0]:
                                self.move_group.execute(retract_plan[1], wait=True)
                                rospy.loginfo("✅ Retracted from key")
                            
                            self.move_group.stop()
                            self.move_group.clear_pose_targets()
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
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False

    def type_key_simple_pose(self, key_name, target_pos):
        try:
            rospy.loginfo(f"🎯 Simple pose approach for '{key_name}'")
            
            current_pose = self.move_group.get_current_pose().pose
            
            target_pose = Pose()
            target_pose.position.x = target_pos[0]
            target_pose.position.y = target_pos[1]
            target_pose.position.z = target_pos[2] + 0.005  
            target_pose.orientation = current_pose.orientation
            
            self.move_group.set_pose_target(target_pose)
            self.move_group.set_planning_time(5.0)
            
            plan = self.move_group.plan()
            
            if plan[0]:
                success = self.move_group.execute(plan[1], wait=True)
                if success:
                    rospy.loginfo(f"✅ Simple pose typing for '{key_name}'")
                    rospy.sleep(self.dwell_time)
                    self.move_group.stop()
                    self.move_group.clear_pose_targets()
                    return True
            
        except Exception as e:
            rospy.logerr(f"Simple pose typing error: {e}")
            
        finally:
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            
        return False

    def get_robot_status(self):
        try:
            status = {
                'timestamp': rospy.Time.now().to_sec(),
                'keyboard_detected': self.keyboard_detected,
                'num_keys_detected': len(self.keypoints_3d),
                'num_saved_keys': len(self.saved_keypoints_3d),
                'coordinate_saving_mode': self.coordinate_saving_mode,
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
            
            # Save coordinates one final time if we have fresh detections
            if self.keypoints_3d:
                rospy.loginfo("💾 Saving coordinates before shutdown...")
                self.save_coordinates()
            
            moveit_commander.roscpp_shutdown()
            
            rospy.loginfo("✅ Shutdown complete")
            
        except Exception as e:
            rospy.logerr(f"Shutdown error: {e}")

def main():
    try:
        controller = RoboticTypingController()
        
        rospy.on_shutdown(controller.shutdown_handler)
        
        rospy.loginfo("🤖 Robotic Typing Controller is ready!")
        rospy.loginfo("📝 Send text to '/type_text' topic to start typing")
        rospy.loginfo("💾 Send message to '/save_coordinates' to save current key positions")
        rospy.loginfo("🔄 Send message to '/toggle_coordinate_saving' to toggle auto-save mode")
        rospy.loginfo("🎨 Watch '/annotated_keyboard' topic for visual feedback")
        rospy.loginfo("   - GREEN keys: Fresh detection with good depth")
        rospy.loginfo("   - YELLOW keys: Using saved coordinates")
        rospy.loginfo("   - RED keys: Poor or no depth data")
        
        rospy.spin()
        
    except KeyboardInterrupt:
        rospy.loginfo("👋 User interrupted")
    except Exception as e:
        rospy.logfatal(f"❌ Fatal error: {e}")
    finally:
        rospy.loginfo("🔄 Exiting...")

if __name__ == "__main__":
    main()