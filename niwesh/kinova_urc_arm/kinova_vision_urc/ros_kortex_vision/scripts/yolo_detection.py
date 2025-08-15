#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch

class YOLOv5ROSNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('yolov5_ros_node', anonymous=True)

        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Subscribe to the camera image topic
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        # Publisher for annotated image (optional)
        self.image_pub = rospy.Publisher("/yolo/detected_image", Image, queue_size=10)

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Perform inference
            results = self.model(cv_image)

            # Render detections on the image
            rendered_image = results.render()[0]

            # Convert the annotated image back to a ROS Image message
            annotated_image_msg = self.bridge.cv2_to_imgmsg(rendered_image, "bgr8")

            # Publish the annotated image (optional)
            self.image_pub.publish(annotated_image_msg)

            # Display the annotated image (optional)
            cv2.imshow("YOLOv5 Detection", rendered_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        yolo_node = YOLOv5ROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass