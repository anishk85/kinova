#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
from ultralytics import YOLO  # type: ignore # YOLOv8 from Ultralytics

class YOLOv8DepthNode:
    def __init__(self):
        rospy.init_node("yolov8_depth_tf_node", anonymous=True)
        self.model = YOLO("yolov8n.pt")  # Load YOLOv8 Nano model
        self.bridge = CvBridge()

        # Camera Intrinsic Parameters
        self.depth_info = {
            "K": [360.01333, 0.0, 243.87228, 0.0, 360.013366699, 137.9218444, 0.0, 0.0, 1.0]
        }

        # ROS Subscribers & Publishers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.image_pub = rospy.Publisher("/yolo/detected_image", Image, queue_size=10)

        # TF2 Buffer & Listener for Transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.depth_image = None


    def depth_callback(self, msg):
        """Convert the depth image message to a NumPy array."""
        try:
            if msg.encoding == "16UC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) * 0.001  # Convert mm to meters
            elif msg.encoding == "32FC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            rospy.logerr(f"Depth processing error: {e}")

    def compute_3d_coordinates(self, u, v, depth):
        """Compute 3D world coordinates from pixel coordinates using camera intrinsics."""
        K = np.array(self.depth_info["K"]).reshape(3, 3)
        uv1 = np.array([u, v, 1.0])
        xyz = np.linalg.inv(K) @ uv1 * depth
        return xyz[0], xyz[1], xyz[2]

    def transform_to_world(self, x, y, z, camera_frame="camera_depth_frame"):
        """Transform the 3D point from the camera frame to the world frame using TF2."""
        try:
            transform = self.tf_buffer.lookup_transform("world", camera_frame, rospy.Time(0), rospy.Duration(1.0))
            point = PointStamped()
            point.header.frame_id = camera_frame
            point.point.x, point.point.y, point.point.z = x, y, z
            transformed_point = do_transform_point(point, transform)
            return transformed_point.point.x, transformed_point.point.y, transformed_point.point.z
        except tf2_ros.LookupException as e:
            rospy.logerr(f"TF Lookup failed: {e}")
            return None

    def image_callback(self, msg):
        """Process the RGB image, run YOLOv8, and compute object depth and world coordinates."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model(cv_image)  # YOLOv8 inference
            rendered_image = cv_image.copy()

            if self.depth_image is not None:
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                        conf = box.conf.item()  # Confidence score
                        cls = box.cls.item()  # Class ID

                        # Compute center of detected object
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Get depth at object center
                        if 0 <= center_x < self.depth_image.shape[1] and 0 <= center_y < self.depth_image.shape[0]:
                            depth_value = self.depth_image[center_y, center_x]
                            if depth_value > 0:
                                x, y, z = self.compute_3d_coordinates(center_x, center_y, depth_value)
                                world_coords = self.transform_to_world(x, y, z)

                                if world_coords:
                                    rospy.loginfo(f"Object ID {cls} at (World): x={world_coords[0]:.2f}, y={world_coords[1]:.2f}, z={world_coords[2]:.2f}")

                                # Draw detection and info
                                self.draw_detections(rendered_image, (x1, y1, x2, y2), center_x, center_y, depth_value, world_coords)
            
            self.publish_and_display(rendered_image)
        
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")

    def draw_detections(self, img, bbox, u, v, depth, world_coords):
        """Draw bounding boxes and display depth/world coordinates."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (u, v), 5, (255, 0, 0), -1)
        cv2.putText(img, f"Depth: {depth:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if world_coords:
            wx, wy, wz = world_coords
            cv2.putText(img, f"World: ({wx:.2f}, {wy:.2f}, {wz:.2f})m", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    def publish_and_display(self, img):
        """Publish processed image and display results."""
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        cv2.imshow("YOLOv8 Detection with Depth", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    YOLOv8DepthNode()
    rospy.spin()
