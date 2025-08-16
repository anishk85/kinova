#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
import message_filters

class Detector:
    def __init__(self):
        rospy.init_node('yolo_detector', anonymous=True)
        self.model = YOLO('trained_yolov8n.pt')

        # Topic names (get from ROS params)
        self.color_image_topic = rospy.get_param('~color_image_topic', '/camera/color/image_raw')
        self.depth_image_topic = rospy.get_param('~depth_image_topic', '/camera/depth/image_raw')
        self.camera_info_topic = rospy.get_param('~camera_info_topic', '/camera/depth/camera_info')  # Use depth camera info
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera_link')  # Frame of the depth camera

        self.bridge = CvBridge()
        # Use message_filters.ApproximateTimeSynchronizer to synchronize the images
        self.color_sub = message_filters.Subscriber(self.color_image_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_image_topic, Image)
        self.info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self.camera_info_callback)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.image_publisher = rospy.Publisher('yolo_output', Image, queue_size=10)
        self.points_publisher = rospy.Publisher('object_positions', PointCloud2, queue_size=10)
        self.depth_image = None
        self.camera_matrix = None

        self.class_names = [
            'accent', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'minus', 'plus', 'del', 'tab', 'q', 'w', 'e', 'r', 't',
            'y', 'u', 'i', 'o', 'p', '[', ']', 'enter', 'caps', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ':',
            '"', '\\', 'shift-left', 'less', 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/', 'shift-right',
            'ctrl-left', 'alt-left', 'space', 'alt-right', 'ctrl-right', 'keyboard'
        ]

        # Use tf2_ros.Buffer instead of BufferClient
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)

    def image_callback(self, color_msg, depth_msg):
        try:
            # Convert ROS Image messages to OpenCV images
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            # Check the encoding of the depth image and convert accordingly
            if depth_msg.encoding == "16UC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1").astype(np.float32) * 0.001
            elif depth_msg.encoding == "32FC1":
                self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            else:
                rospy.logerr(f"Unsupported depth image encoding: {depth_msg.encoding}")
                return
            img = cv2.resize(color_image, (640, 640))  # Resize image for YOLO
            img_copy = img.copy()

            if self.camera_matrix is None or self.depth_image is None:
                rospy.logwarn("Camera info or depth image not yet received.")
                return

            results = self.model(img, iou=0.7, conf=0.5)
            results = results[0]
            object_points = []  # List to store 3D points for detected objects

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    label = self.class_names[class_id]
                    confidence = box.conf[0]

                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{label}', (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                    # Estimate 3D position (center of bounding box)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Check if the center point is within the bounds of the depth image
                    if 0 <= center_x < self.depth_image.shape[1] and 0 <= center_y < self.depth_image.shape[0]:
                        depth = self.depth_image[center_y, center_x]

                        if not np.isnan(depth) and not np.isinf(depth) and depth > 0:  # Handle potential NaN/Inf depths
                            # Backproject to 3D
                            point_x = (center_x - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
                            point_y = (center_y - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
                            point_z = depth
                            #corners of bounding boxes
                            

                            # Transform to base frame
                            try:
                                transform = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame,
                                                                           rospy.Time(0), rospy.Duration(0.1))
                                point_msg = tf2_ros.ConvertPointStamped.from_xyz(point_x, point_y, point_z,
                                                                                 rospy.Time.now(),
                                                                                 self.camera_frame)
                                transformed_point = tf2_ros.do_transform_point(point_msg, transform)
                                object_points.append(
                                    [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z])

                            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                                    tf2_ros.ExtrapolationException) as e:
                                rospy.logwarn(f"TF error: {e}")
                                object_points.append([point_x, point_y, point_z])

                        else:
                            rospy.logwarn("Invalid depth value.")
                    else:
                        rospy.logwarn("Center point outside depth image bounds.")

            # Publish the image with bounding boxes
            self.image_publisher.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))

            # Publish 3D point cloud
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = self.base_frame
            if object_points:
                point_cloud_msg = pc2.create_cloud_xyz32(header, object_points)
                self.points_publisher.publish(point_cloud_msg)
                
            # Real time display with bounding boxes
            cv2.imshow("YOLOv5", img)
            cv2.waitKey(1)


        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    detector = Detector()
    detector.run()