#!/usr/bin/env python

import os
import sys
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PointStamped
import message_filters


class Projector:
    """
    """
    def __init__(self,
                 depth_image_topic="/camera/depth/image_raw",
                 depth_info_topic = "/camera/depth/camera_info"
                 ):
        """
        """
        queue_size = 1
        self.rospy_rate = rospy.Rate(5)

        self.depth_sub = None


        self.image_sub = message_filters.Subscriber(depth_image_topic, Image)
        self.info_sub = message_filters.Subscriber(depth_info_topic, CameraInfo)
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.info_sub], 10)
        self.ts.registerCallback(self.__callback)

        # self.depth_sub = rospy.Subscriber(depth_image_topic,
        #                                   Image,
        #                                   self.__depth_image_cb,
        #                                   queue_size = queue_size)

        # self.info_sub = rospy.Subscriber(info_image_topic,
        #                                  CameraInfo,
        #                                  self.__depth_image_cb,
        #                                  queue_size = queue_size)

        self.pub = rospy.Publisher("/mypoint", PointStamped, queue_size=10)


    def __callback(self, image_msg, info_msg):
        """
        """
        print("{} {}".format(image_msg.header.stamp, info_msg.header.stamp))
        image = CvBridge().imgmsg_to_cv2(image_msg, "passthrough")

        omsg = PointStamped()
        omsg.header = image_msg.header
        h = 340
        w = 272
        d = float(image[h][w])
        print(info_msg.K)
        # Intrinsic camera matrix for the raw (distorted) images.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]
        # Projects 3D points in the camera coordinate frame to 2D pixel
        # coordinates using the focal lengths (fx, fy) and principal point
        # (cx, cy).
        fx, _, cx, _, fy, cy, _, _, _ = info_msg.K
        print("fx {}".format(fx))
        print("fy {}".format(fy))
        print("cx {}".format(cx))
        print("cy {}".format(cy))
        print(d)

        x = (w - cx) * image[h][w] / fx
        y = (h - cy) * image[h][w] / fy
        z = image[h][w]
        print("x {}".format(x))
        omsg.point.x = x/1000.0
        omsg.point.y = y/1000.0
        omsg.point.z = d/1000.0
        print(omsg)
        self.pub.publish(omsg)
        # cv2.imshow("Person Manager", image)
        # cv2.waitKey(1)



    def __depth_image_cb(self, imsg):
        """
        """
        # rospy.loginfo("Depth at {}".format(imsg.header.stamp))
        omsg = PointStamped()
        omsg.header = imsg.header
        image = CvBridge().imgmsg_to_cv2(imsg, "passthrough")
        d = float(image[272][240])/1000.0
        print(d)
        omsg.point.z = d
        self.pub.publish(omsg)


    def run(self):
        """
        """
        while not rospy.is_shutdown():
            # rospy.loginfo("running")
            self.rospy_rate.sleep()


if __name__ == "__main__":
    """
    """
    color_image_topic = "/camera/rgb/image_raw"
    depth_image_topic = "/camera/depth/image_raw"

    if len(sys.argv) > 1: depth_image_topic = sys.argv[1]
    if len(sys.argv) > 2: depth_info_topic = sys.argv[2]

    try:
        rospy.init_node("projector", anonymous=False)
        p = Projector(depth_image_topic=depth_image_topic,
                      depth_info_topic=depth_info_topic)
        p.run()
    except rospy.ROSInterruptException:
        pass
