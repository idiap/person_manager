#!/usr/bin/env python
# coding: utf8

######################################################################
# Copyright (c) 2018 Idiap Research Institute <http://www.idiap.ch/>
######################################################################

import os
import sys
import time
import copy
import random
import logging
import argparse

from collections import deque

import numpy as np
import cv2
import rospy

from sensor_msgs.msg import JointState
from perception_msgs.msg import Hold

WINDOW_NAME = "Joints"

class PerceptionHolder(object):
    """
    """
    def __init__(self,
                 joint_topic,
                 joints,
                 threshold=0.1,
                 beta=0.9,
                 rate=30,
                 max_length_buffer=1000):
        """
        """
        self.rate = rate
        self.rospy_rate = rospy.Rate(self.rate)

        self.joints_of_interest = joints.split(",")
        self.indices = [] # Array index where the joint is in msg.name

        self.joints = [] # All the names from the msg

        self.threshold = threshold
        self.beta = beta

        self.height = 480
        self.max_length_buffer = max_length_buffer
        self.joint_buffer = deque(maxlen=self.max_length_buffer)

        self.prev_values = None
        self.prev_diff = None

        self.hold = False

        queue_size = 10
        self.joint_sub = rospy.Subscriber(joint_topic,
                                          JointState,
                                          self._joints_cb,
                                          queue_size=queue_size)

        self.hold_pub = rospy.Publisher("hold_perception",
                                        Hold,
                                        queue_size=queue_size)

    def _joints_cb(self, imsg):
        """
        """
        # ns = imsg.header.stamp.to_nsec()
        # self.joint_buffer.append((ns, dict(zip(imsg.name, imsg.position))))


        # Store at which index the joint is in the message
        if len(self.indices) == 0:
            self.joints = imsg.name
            for i, joint in enumerate(self.joints_of_interest):
                for j, name in enumerate(imsg.name):
                    if joint == name:
                        self.indices.append(j)
                        break

            rospy.loginfo("Base filtering on {} at {}". \
                          format(self.joints_of_interest, self.indices))

        current_values = np.array(imsg.position, dtype=np.float)

        if self.prev_values is None:
            self.prev_values = current_values
            self.prev_diff = np.zeros_like(current_values)

        current_diff = current_values - self.prev_values
        self.prev_values = current_values

        current_diff = self.beta*self.prev_diff + (1.0 - self.beta)*current_diff
        self.prev_diff = current_diff

        # igreater = np.argwhere(current_diff>self.threshold)
        # greater = np.where(current_diff>self.threshold, current_diff, 0)
        hold = False
        for i in self.indices:
            if current_diff[i] > self.threshold:
                hold = True
                rospy.loginfo("{} is > threshold". \
                              format(self.joints_of_interest[i]))

        self.hold = hold

        # print("hold {}".format(hold))

    def run(self):
        """
        """
        # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        fields = ["HeadYaw", "HeadPitch"]

        while not rospy.is_shutdown():
            omsg = Hold()
            omsg.hold = self.hold
            # print("omsg {}".format(omsg))
            self.hold_pub.publish(omsg)

            # data = list(copy.deepcopy(self.joint_buffer))
            # data = sorted(data, key=lambda x: x[0])

            # # pts = [ [] for c in range(len(fields))]
            # pts = {}
            # for i, field in enumerate(fields):
            #     pts[field] = [(j, self.height/2.0 + 50*x[1][field]) for j, x in enumerate(data)]

            #     derivative = []
            #     for j in range(len(pts[field])):
            #         if j == 0:
            #             derivative.append((pts[field][j][1], 0))
            #         else:
            #             diff = pts[field][j][1] - pts[field][j-1][1]
            #             derivative.append((pts[field][j][0], self.height/2.0 + 50*diff))
            #     pts["d{}".format(field)] = derivative

            # # line = np.int32(line)
            # image = np.full((self.height, self.max_length_buffer, 3), 255).astype(np.uint8)

            # for i, field in enumerate(pts):
            #     color = (( (i % 8) // 4),
            #              ( (i % 4) // 2),
            #              ( (i % 2) ))
            #     cv2.polylines(image, np.int32([pts[field]]), 0, [255*x for x in color])


            # cv2.imshow(WINDOW_NAME, image)
            # cv2.waitKey(30)
            # # print([t for (t, _) in data])
            # # print(self.joint_buffer)
            time.sleep(1.0/self.rate)


if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic",
                        type=str,
                        default="/joint_states",
                        help="sensor_msgs/JointState")
    parser.add_argument("-t", "--threshold",
                        type=float,
                        default=0.01,
                        help="Value above which perception is held (speed of joints)")
    parser.add_argument("-j", "--joints",
                        type=str,
                        default="HeadYaw,HeadPitch",
                        help="Which joints to monitor")
    parser.add_argument("--pwd", type=str,
                        default="/tmp",
                        help="Path of working directory")
    parser.add_argument("--verbose", type=int,
                        default=20,
                        help="Logging verbosity (10, 20, 30)")
    # parser.add_argument("args", nargs=argparse.REMAINDER)

    try:
        args = parser.parse_args(rospy.myargv()[1:])
    except:
        parser.print_help()
        sys.exit(1)

    try:
        rospy.init_node("hold_perception_node")
        rospy.loginfo("Starting hold_perception_node")
        ph = PerceptionHolder(args.topic,
                              args.joints,
                              threshold=args.threshold)
        ph.run()
    except rospy.ROSInterruptException:
        pass
