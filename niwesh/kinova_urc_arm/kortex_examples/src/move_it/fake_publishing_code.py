#!/usr/bin/env python3

import rospy
import random
from geometry_msgs.msg import PointStamped

def publish_test_data():
    rospy.init_node('test_yolo_publisher', anonymous=True)
    pub = rospy.Publisher('/yolo/world_coordinates', PointStamped, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    base_x, base_y, base_z = 1.5, 0.5, 0.5  # Base position

    while not rospy.is_shutdown():
        # Simulate occasional missing data (30% chance to skip publishing)
        if random.random() < 0.2:
            rospy.logwarn("Skipping this iteration (no data published)")
        else:
            msg = PointStamped()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "world"
            
            # Add small random deviations (±0.05) to each coordinate
            msg.point.x = base_x + random.uniform(-0.01, 0.01)
            msg.point.y = base_y + random.uniform(-0.001, 0.001)
            msg.point.z = base_z + random.uniform(-0.001, 0.001)

            rospy.loginfo(f"Publishing: x={msg.point.x:.3f}, y={msg.point.y:.3f}, z={msg.point.z:.3f}")
            pub.publish(msg)

        rate.sleep()

if __name__ == "__main__":
    try:
        publish_test_data()
    except rospy.ROSInterruptException:
        pass
