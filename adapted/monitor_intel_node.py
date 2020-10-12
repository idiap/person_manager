#!/usr/bin/env python
# coding=utf-8

######################################################################
# This script should be put on the Jetson in /home/nvidia
######################################################################

import os
import sys
import cv2
import time
import copy
import argparse
import datetime

import rospy
import rosgraph

from sensor_msgs.msg import CameraInfo


class IntelMonitor(object):
    """
    """
    def __init__(self,
                 timeout=5,
                 rate=1,
                 color_topic="/camera/color/camera_info"):
        """
        Args:
            timeout: How many seconds without any image messages to wait before rebooting the Jetson

        """
        self.timeout = timeout
        self.last_time_img_received = None
        self.rate = rospy.Rate(rate)

        self.color_sub = rospy.Subscriber(color_topic,
                                          CameraInfo,
                                          self.__color_image_cb,
                                          queue_size=10)


    def __color_image_cb(self, imsg):
        """
        """
        self.last_time_img_received = datetime.datetime.now()

    def _reboot(self):
        """
        """
        rospy.loginfo("Rebooting the Jetson")
        # os.system("pwd; ls")
        os.system('sudo reboot')

    def run(self):
        while not rospy.is_shutdown():
            if self.last_time_img_received is not None:
                now = datetime.datetime.now()
                seconds_since_last = (now - self.last_time_img_received).total_seconds()

                # Check if camera is on
                if seconds_since_last > self.timeout:
                    self._reboot()
                    # Set to None not to reboot continuously
                    self.last_time_img_received = None
                else:
                    rospy.loginfo("Last image was {} s ago < {} s"
                                  .format(seconds_since_last, self.timeout))

                # Check if roscore is running
                pid = None
                try:
                    pid = rosgraph.Master('/rostopic').getPid()
                    rospy.loginfo("roscore pid is {}".format(pid))
                except:
                    rospy.loginfo("Unable to communicate with roscore master")
                    self._reboot()

            else:
                rospy.loginfo("last_time_img_received is None")

            self.rate.sleep()


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()

    try:
        opts = parser.parse_args(rospy.myargv()[1:])
    except:
        parser.print_help()
        sys.exit(1)

    try:
        rospy.init_node("monitor_intel_node", anonymous=False)
        rospy.loginfo("Starting monitoring Intel camera")
        m = IntelMonitor()
        m.run()
    except rospy.ROSInterruptException:
        pass
