#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
import cv2
# import torch
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from geometry_msgs.msg import PointStamped

# Global variables
model = None
bridge = None
label_id = 39  # 39 is the label ID for the bottle class
depth_image = None
tf_buffer = None
tf_listener = None
image_pub = None
world_coords_pub = None  # Publisher for world coordinates



# Camera parameters
color_info = {
    'K': [1297.672904, 0.0, 620.914026, 0.0, 1298.631344, 238.280325, 0.0, 0.0, 1.0]
}

depth_info = {
    'K': [360.01333, 0.0, 243.87228, 0.0, 360.013366699, 137.9218444, 0.0, 0.0, 1.0]
}

def init_node():
    """Initialize the ROS node and global variables."""
    global model, bridge, label_id, tf_buffer, tf_listener, image_pub, world_coords_pub

    rospy.init_node('yolov8_ros_node', anonymous=True)

    # Load YOLOv8 Nano model
    model = YOLO('yolov8n.pt')
    bridge = CvBridge()

    # Get label ID parameter
    label_id = rospy.get_param('~label_id', default=None)  # Default to bottle
    rospy.loginfo(f"Filtering detections for label ID: {label_id}")

    # Initialize TF2
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    # Set up publishers
    image_pub = rospy.Publisher("/yolo/detected_image", Image, queue_size=10)
    world_coords_pub = rospy.Publisher("/yolo/world_coordinates", PointStamped, queue_size=10)

    # Set up subscribers
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    rospy.Subscriber("/camera/depth/image_raw", Image, depth_callback)


def depth_callback(msg):
    """Process depth image."""
    global depth_image
    try:
        if msg.encoding == "16UC1":
            depth_image = bridge.imgmsg_to_cv2(msg, "16UC1").astype(np.float32) * 0.001
        elif msg.encoding == "32FC1":
            depth_image = bridge.imgmsg_to_cv2(msg, "32FC1")
    except Exception as e:
        rospy.logerr(f"Depth processing error: {e}")


def compute_3d_coordinates(u, v, depth):
    """Compute 3D coordinates from depth."""
    if depth <= 0 or np.isnan(depth) or np.isinf(depth):
        return None

    K = np.array(depth_info['K']).reshape(3, 3)
    uv1 = np.array([u, v, 1.0])
    xyz = np.linalg.inv(K) @ uv1 * depth
    return xyz[0], xyz[1], xyz[2]

def transform_point(camera_coords):
    """Transform a point from the camera frame to the world frame."""
    try:
        transform = tf_buffer.lookup_transform("base_link", "camera_depth_frame", rospy.Time(0), rospy.Duration(1.0))

        point_stamped = PointStamped()
        point_stamped.header.frame_id = "camera_depth_frame"
        point_stamped.header.stamp = rospy.Time.now()
        point_stamped.point.x, point_stamped.point.y, point_stamped.point.z = camera_coords

        world_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
        return world_point.point.x, world_point.point.y, world_point.point.z
    except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
        return None


def publish_world_coords(world_coords):
    """Publish world coordinates to the topic."""
    if world_coords:
        point_msg = PointStamped()
        point_msg.header.frame_id = "base_link"
        point_msg.header.stamp = rospy.Time.now()
        point_msg.point.x, point_msg.point.y, point_msg.point.z = world_coords

        world_coords_pub.publish(point_msg)


def draw_detections(image, bbox, center, world_coords, class_label, depth_value=None):
    """Draw bounding boxes and display object coordinates."""
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.circle(image, center, 5, (255, 0, 0), -1)

    cv2.putText(image, f"{class_label}", (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if depth_value is not None:
        cv2.putText(image, f"Depth (cam): {depth_value:.2f} m",
                (bbox[0], bbox[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 165, 255), 3)   # bigger font + bold


    if world_coords:
        x_world, y_world, z_world = world_coords
        cv2.putText(image, f"World: ({x_world:.2f}, {y_world:.2f}, {z_world:.2f})m",
                    (bbox[0], bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        rospy.loginfo(f"World coordinates not available for {class_label}")


def publish_and_display(rendered_image):
    """Publish the processed image and display it."""
    try:
        image_msg = bridge.cv2_to_imgmsg(rendered_image, encoding="bgr8")
        image_pub.publish(image_msg)
        cv2.imshow("YOLO Detections", rendered_image)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logerr(f"Error publishing/displaying image: {e}")

def image_callback(msg):
    """Process the color image and apply YOLOv8."""
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        results = model(cv_image)  # YOLOv8 inference
        rendered_image = cv_image.copy()

        if depth_image is not None:
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])

                    # Filter detections based on the specified label ID
                    if label_id is not None and class_id != label_id:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_rgb = ((x1 + x2) // 2, (y1 + y2) // 2)

                    # Project RGB center to depth frame
                    u_depth = int(((center_rgb[0] - color_info['K'][2]) *
                                   (depth_info['K'][0] / color_info['K'][0])) +
                                  depth_info['K'][2])
                    v_depth = int(((center_rgb[1] - color_info['K'][5]) *
                                   (depth_info['K'][4] / color_info['K'][4])) +
                                  depth_info['K'][5])

                    world_coords = None

                    if 0 <= u_depth < depth_image.shape[1] and 0 <= v_depth < depth_image.shape[0]:
                        depth_value = depth_image[v_depth, u_depth]

                        if not np.isnan(depth_value) and not np.isinf(depth_value):
                            camera_coords = compute_3d_coordinates(u_depth, v_depth, depth_value)
                            if camera_coords:
                                world_coords = transform_point(camera_coords)

                            # rospy.loginfo(f"[YOLO] {class_label} | Depth (camera): {depth_value:.2f} m")

                    # Publish world coordinates
                    if world_coords:
                        publish_world_coords(world_coords)

                    # Get class label and draw detections
                    class_label = model.names[class_id]
                    #draw_detections(rendered_image, (x1, y1, x2, y2), center_rgb, world_coords, class_label)
                    draw_detections(rendered_image, (x1, y1, x2, y2), center_rgb, world_coords, class_label, depth_value)

        publish_and_display(rendered_image)

    except Exception as e:
        rospy.logerr(f"Image processing error: {e}")

def main():
    """Main function to initialize and run the node."""
    init_node()
    rospy.spin()

if __name__ == "__main__":
    main()
