#!/usr/bin/env python
# coding: utf8

"""Publish perception_msgs as RViz markers"""

import os
import sys
import cv2
import time
import copy
import argparse

import rospy
from cv_bridge import CvBridge

from collections import deque

from std_msgs.msg import ColorRGBA

from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage


from perception_msgs.msg import TrackedPersonArray
from perception_msgs.msg import VoiceActivityArray
from perception_msgs.msg import GazeInfoArray

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from utils import get_color
from utils import draw_bounding_box
from utils import draw_probabilities
from utils import scale_bounding_box
from utils import get_closest_rosmsg
from utils import draw_body_joints_2d
from utils import draw_facial_landmarks

import message_filters

WINDOW_DISPLAY = "Idiap Perception"

def new_head_marker(p):
    """
    """
    m = Marker()
    # m.type = m.CUBE
    m.type = m.ARROW
    # m.scale = Vector3(0.2, 0.2, 0.2)
    m.scale = Vector3(0.2, 0.05, 0.05)
    m.color = ColorRGBA(1, 0, 0, 0.5)
    m.pose = p.head_pose
    m.lifetime = rospy.Duration(0.25)
    return m

def new_body_marker(p):
    """
    """
    m = Marker()
    m.type = m.SPHERE_LIST
    m.scale = Vector3(0.2, 0.2, 0.2)
    m.lifetime = rospy.Duration(0.25)
    m.scale = Vector3(0.07, 0.07, 0.07)

    if p.body_joints_3d.leye != Point(0,0,0):
        m.points.append(p.body_joints_3d.leye)
        m.colors.append(ColorRGBA(0, 1, 0, 1))
    if p.body_joints_3d.reye != Point(0,0,0):
        m.points.append(p.body_joints_3d.reye)
        m.colors.append(ColorRGBA(0, 1, 0, 1))

    if p.body_joints_3d.lear != Point(0,0,0):
        m.points.append(p.body_joints_3d.lear)
        m.colors.append(ColorRGBA(0, 1, 1, 1))
    if p.body_joints_3d.rear != Point(0,0,0):
        m.points.append(p.body_joints_3d.rear)
        m.colors.append(ColorRGBA(0, 1, 1, 1))

    if p.body_joints_3d.nose != Point(0,0,0):
        m.points.append(p.body_joints_3d.nose)
        m.colors.append(ColorRGBA(1, 0, 0, 1))
    if p.body_joints_3d.neck != Point(0,0,0):
        m.points.append(p.body_joints_3d.neck)
        m.colors.append(ColorRGBA(1, 0, 0, 1))

    if p.body_joints_3d.lshoulder != Point(0,0,0):
        m.points.append(p.body_joints_3d.lshoulder)
        m.colors.append(ColorRGBA(1, 0, 1, 1))
    if p.body_joints_3d.rshoulder != Point(0,0,0):
        m.points.append(p.body_joints_3d.rshoulder)
        m.colors.append(ColorRGBA(1, 0, 1, 1))

    if p.body_joints_3d.relbow != Point(0,0,0):
        m.points.append(p.body_joints_3d.relbow)
        m.colors.append(ColorRGBA(1, 0, 1, 1))
    if p.body_joints_3d.lelbow != Point(0,0,0):
        m.points.append(p.body_joints_3d.lelbow)
        m.colors.append(ColorRGBA(1, 0, 1, 1))

    if p.body_joints_3d.rwrist != Point(0,0,0):
        m.points.append(p.body_joints_3d.rwrist)
        m.colors.append(ColorRGBA(1, 0, 1, 1))
    if p.body_joints_3d.lwrist != Point(0,0,0):
        m.points.append(p.body_joints_3d.lwrist)
        m.colors.append(ColorRGBA(1, 0, 1, 1))

    if p.body_joints_3d.rhip != Point(0,0,0):
        m.points.append(p.body_joints_3d.rhip)
        m.colors.append(ColorRGBA(1, 0, 1, 1))
    if p.body_joints_3d.lhip != Point(0,0,0):
        m.points.append(p.body_joints_3d.lhip)
        m.colors.append(ColorRGBA(1, 0, 1, 1))

    if p.body_joints_3d.rknee != Point(0,0,0):
        m.points.append(p.body_joints_3d.rknee)
        m.colors.append(ColorRGBA(1, 0, 1, 1))
    if p.body_joints_3d.lknee != Point(0,0,0):
        m.points.append(p.body_joints_3d.lknee)
        m.colors.append(ColorRGBA(1, 0, 1, 1))

    if p.body_joints_3d.rankle != Point(0,0,0):
        m.points.append(p.body_joints_3d.rankle)
        m.colors.append(ColorRGBA(1, 0, 1, 1))
    if p.body_joints_3d.lankle != Point(0,0,0):
        m.points.append(p.body_joints_3d.lankle)
        m.colors.append(ColorRGBA(1, 0, 1, 1))

    return m

class Visualiser(object):
    """
    """
    def __init__(self,
                 color_topic="/naoqi_driver_node/camera/front/image_raw",
                 track_topic="/wp2/track",
                 voice_topic="/wp2/voice",
                 gaze_topic="/wp2/gaze",
                 maxlen=50,
                 rospy_rate=30,
                 visu=0,
                 markers=0,
                 record=""):
        """
        """
        queue_size = 10

        self.maxlen = maxlen
        self.rospy_rate = rospy.Rate(rospy_rate)

        self.color_buffer = deque(maxlen=self.maxlen)
        self.track_buffer = deque(maxlen=self.maxlen)
        self.voice_buffer = deque(maxlen=self.maxlen)
        self.gaze_buffer = deque(maxlen=self.maxlen)

        self.cvbridge = CvBridge()

        self.visu = visu
        self.markers = markers

        self.record = record
        if len(record) > 0:
            if not os.path.exists(record):
                os.makedirs(record)

        # If record is not "", or if visu is turned on, we need to
        # generate the visu image.
        self.do_visu = 0
        if len(record) > 0 or visu > 0:
            self.do_visu = 1

        # Keep trace of who speaking to change display accordingly
        # (is_speaking comes in a different topic than the location of
        # people)
        self.is_speaking = set([])

        # IDs of people with VoiceActivity.turn_duration > 0
        self.has_turn = set([])

        rospy.loginfo("Subscribe to {}".format(color_topic))
        self.color_sub = None
        self.use_compressed_color = False
        if color_topic.endswith("compressed"):
            self.color_sub = rospy.Subscriber(color_topic,
                                              CompressedImage,
                                              self.__color_image_cb,
                                              queue_size=10)
            self.use_compressed_color = True
        else:
            self.color_sub = rospy.Subscriber(color_topic,
                                              Image,
                                              self.__color_image_cb,
                                              queue_size=10)
        # self.color_sub = rospy.Subscriber(color_topic,
        #                                   Image,
        #                                   self.__color_image_cb,
        #                                   queue_size=queue_size)

        rospy.loginfo("Subscribe to {}".format(track_topic))
        self.track_sub = rospy.Subscriber(track_topic,
                                          TrackedPersonArray,
                                          self.__track_cb,
                                          queue_size=queue_size)

        rospy.loginfo("Subscribe to {}".format(voice_topic))
        self.voice_activity_sub = rospy.Subscriber(voice_topic,
                                                   VoiceActivityArray,
                                                   self.__voice_cb,
                                                   queue_size=queue_size)

        rospy.loginfo("Subscribe to {}".format(gaze_topic))
        self.gaze_activity_sub = rospy.Subscriber(gaze_topic,
                                                   GazeInfoArray,
                                                   self.__gaze_cb,
                                                   queue_size=queue_size)

        self.head_pose_pub = rospy.Publisher("/wp2/head_viz",
                                             MarkerArray,
                                             queue_size=10)

        self.body_joint_pub = rospy.Publisher("/wp2/body_viz",
                                              MarkerArray,
                                              queue_size=10)



    def __color_image_cb(self, imsg):
        """
        """
        self.color_buffer.append(copy.deepcopy(imsg))


    def __voice_cb(self, imsg):
        """
        """
        self.voice_buffer.append(copy.deepcopy(imsg))

        self.is_speaking.clear()
        self.has_turn.clear()
        for m in imsg.data:
            if m.is_speaking:
                self.is_speaking.add(m.person_id)
            if m.turn_duration > rospy.Duration(0):
                self.has_turn.add(m.person_id)


    def __gaze_cb(self, imsg):
        """
        """
        self.gaze_buffer.append(copy.deepcopy(imsg))


    def __track_cb(self, imsg):
        """
        """
        self.track_buffer.append(copy.deepcopy(imsg))

        if self.markers > 0:
            head_msg = MarkerArray()
            body_msg = MarkerArray()

            idx = 0
            for i,p in enumerate(imsg.data):
                # Head pose
                h = new_head_marker(p)
                h.header = imsg.header
                h.id = idx
                if p.person_id in self.is_speaking:
                    h.type = h.SPHERE
                    h.color = ColorRGBA(0, 1, 0, 0.5)
                    h.scale = Vector3(0.25, 0.25, 0.25)
                head_msg.markers.append(h)
                idx += 1

                # Body joints
                b = new_body_marker(p)
                b.header = imsg.header
                b.id = idx
                body_msg.markers.append(b)
                idx += 1

            self.head_pose_pub.publish(head_msg)
            self.body_joint_pub.publish(body_msg)


    def draw_visu(self, tns):
        """Draw the tracker output on 2D image at time 'tns'

        Args:
            tns  : Time in nano seconds

        """
        image_msg, _ = get_closest_rosmsg(tns, self.color_buffer)
        track_msg, _ = get_closest_rosmsg(tns, self.track_buffer)

        if image_msg is None:
            return None

        if self.use_compressed_color:
            display = self.cvbridge.compressed_imgmsg_to_cv2(image_msg, "bgr8").copy()
        else:
            display = self.cvbridge.imgmsg_to_cv2(image_msg, "bgr8").copy()

        # display = self.cvbridge.imgmsg_to_cv2(image_msg, "bgr8").copy()

        if track_msg is None:
            cv2.putText(display,
                        "No perception message received",
                        (0, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=3)
            return display

        for p in track_msg.data:
            color = get_color(p.person_id)
            draw_body_joints_2d(display, p.body_joints_2d, color)

            x0, x1, y0, y1 = scale_bounding_box(display,
                                                p.box.h, p.box.w,
                                                p.box.height, p.box.width,
                                                p.box.image_height,
                                                p.box.image_width)


            shape = "rect"
            thickness = 3
            if p.person_id in self.is_speaking:
                shape = "circle"
                thickness = 20

            draw_bounding_box(display, x0, y0, x1, y1,
                              color=color,
                              ID=p.person_id,
                              name="",
                              shape=shape,
                              thickness=thickness)

            gaze_msg, _ = get_closest_rosmsg(tns, self.gaze_buffer)
            probabilities = {}
            for v in gaze_msg.data:
                if v.person_id == p.person_id:
                    probabilities = { a.target_id:a.probability for a in v.attentions }
                    break

            draw_probabilities(display, probabilities, x0, y0, x1, y1)


            voice_msg, _ = get_closest_rosmsg(tns, self.voice_buffer)
            probabilities = {}
            for v in voice_msg.data:
                if v.person_id == p.person_id:
                    probabilities = { a.target_id:a.probability for a in v.addresse }
                    break

            if len(probabilities) > 0:
                draw_probabilities(display, probabilities, x0, y0, x1, y1 + 7)



            # voice_msg, _ = get_closest_rosmsg(tns, self.voice_buffer)
            # probabilities = {}
            # for v in voice_msg.data:
            #     if v.person_id == p.person_id:
            #         probabilities = { a.target_id:a.probability for a in v.addresse }
            #         break

            # draw_probabilities(display, probabilities, x0, y0, x1, y1)

            # if len(self.voice_buffer) > 0:
            #     for vmsg in self.voice_buffer[-1].data:
            #         print(vmsg.addresse)

        return display


    def choose_latest_timestamp(self):
        """Return the oldeset timestamp between all latest messages"""
        if len(self.track_buffer) > 0:
            return self.track_buffer[-1].header.stamp.to_nsec()
        elif len(self.color_buffer) > 0:
            return self.color_buffer[-1].header.stamp.to_nsec()
        else:
            return 0


    def run(self):
        """
        """
        last_ts_processed = 0
        cv2.namedWindow(WINDOW_DISPLAY, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(WINDOW_DISPLAY, 4*480, 4*260)
        # cv2.moveWindow(WINDOW_DISPLAY, 0, 4*260)

        while not rospy.is_shutdown():
            tns = self.choose_latest_timestamp()
            if last_ts_processed == tns: continue
            last_ts_processed = tns

            if self.do_visu > 0:
                display = self.draw_visu(tns)

                if display is None: continue

                if self.visu > 0:
                    cv2.imshow(WINDOW_DISPLAY, display)
                    cv2.waitKey(30)

                if len(self.record) > 0:
                    nm = "{}.jpg".format(tns)
                    image_path = os.path.join(self.record, nm)
                    cv2.imwrite(image_path, display)


            self.rospy_rate.sleep()

if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--color",
                        type=str,
                        default="/naoqi_driver_node/camera/front/image_raw",
                        help="Registered color topic")
    parser.add_argument("--track",
                        type=str,
                        default="/wp2/track",
                        help="Topic name")
    parser.add_argument("--voice",
                        type=str,
                        default="/wp2/voice",
                        help="Topic name")
    parser.add_argument("--gaze",
                        type=str,
                        default="/wp2/gaze",
                        help="Topic name")
    parser.add_argument("--buffer-length",
                        type=int,
                        default=50,
                        help="Number of message to keep in memory")
    parser.add_argument("--visu",
                        type=int,
                        default=0,
                        help="Whether to display 2D visualisation")
    parser.add_argument("--markers",
                        type=int,
                        default=0,
                        help="Whether to publish RViz markers")
    parser.add_argument("--record",
                        type=str,
                        default="",
                        help="Whether to save the visu to image files")

    try:
        opts = parser.parse_args(rospy.myargv()[1:])
    except:
        parser.print_help()
        sys.exit(1)

    try:
        rospy.init_node("visualisation_node", anonymous=False)
        rospy.loginfo("Starting visualisation")
        v = Visualiser(color_topic=opts.color,
                       track_topic=opts.track,
                       voice_topic=opts.voice,
                       gaze_topic=opts.gaze,
                       maxlen=opts.buffer_length,
                       visu=opts.visu,
                       markers=opts.markers,
                       record=opts.record)
        v.run()
    except rospy.ROSInterruptException:
        pass
