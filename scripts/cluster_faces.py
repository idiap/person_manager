#!/usr/bin/env python
# coding=utf-8

######################################################################
# Copyright (c) 2017 Idiap Research Institute, <http://www.idiap.ch>
######################################################################

from __future__ import print_function

description = "Cluster face based on openface"

import os
import sys
import time
import argparse

import rospy
import rosbag

from cv_bridge       import CvBridge
from sensor_msgs.msg import Image
from perception_msgs.msg import PersonTrackletArray

from pprint import pprint

from align_dlib import AlignDlib

import cv2

import numpy as np
import dlib

def main(bag_file):
    """
    """
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    topics = bag.get_type_and_topic_info()
    # pprint(topics)

    align = AlignDlib("/scratch/ocanevet/models/dlib/shape_predictor_68_face_landmarks.dat")

    msgs = {}

    for topic, msg, t in bag.read_messages(["/front/image_raw", "/tracklet"]):
        t_nsec = msg.header.stamp.to_nsec()
        if t_nsec not in msgs: msgs[t_nsec] = {}

        if topic == "/front/image_raw":
            msgs[t_nsec]["image"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if topic == "/tracklet":
            msgs[t_nsec]["track"] = msg.data

    for t in sorted(msgs.keys()):
        print(t)
        if not "image" in msgs[t]:
            continue

        render = msgs[t]["image"].copy()
        img = msgs[t]["image"].copy()
        # img = cv2.resize(img, dsize=(0,0), fx=2.0, fy=2.0)
        # bbs = align.getAllFaceBoundingBoxes(img)
        # print(len(bbs))

        alignedFaces = []
        # for box in bbs:
            # alignedFaces.append(align.align(96,
            #                                 img,
            #                                 box,
            #                                 landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE))

            # cropalign.align(96, img)
            # cv2.imshow("crop", alignedFaces[-1])

        if "track" in msgs[t]:
            for p in msgs[t]["track"]:
                bb = p.box
                if bb.height < 35: continue

                x0 = int(max(bb.w, 0))
                y0 = int(max(bb.h, 0))
                x1 = int(min((bb.w + bb.width), img.shape[1]-1))
                y1 = int(min((bb.h + bb.height), img.shape[0]-1))
                crop = img[y0:y1,x0:x1]
                cv2.imshow("crop", crop)
                # time.sleep(0.1)
                ali = align.align(96, img,
                                  dlib.rectangle(left=x0, top=y0, right=x1, bottom=y1))
                if ali is not None:
                    print(ali.shape)
                    cv2.imshow("aligned", ali)
                else:
                    print("none" + "*"*70)

                # cropalign.align(96, img)
                cv2.rectangle(render,
                              (bb.w, bb.h),
                              (bb.w + bb.width, bb.h + bb.height),
                              (0,0,255), 3)



        cv2.imshow("image", render)
        cv2.waitKey(1)
        time.sleep(0.1)

    # pprint(msgs)



if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser \
             (description=description,
              add_help=False,
              formatter_class=argparse.RawDescriptionHelpFormatter)


    parser.add_argument("args", nargs=argparse.REMAINDER)

    try:
        opts = parser.parse_args();
    except:
        parser.print_help();
        sys.exit(1);

    print("opts {}".format(opts))

    if len(opts.args) > 0:
        main(opts.args[0])
