#!/usr/bin/env python
# coding: utf8

######################################################################
# Copyright (c) 2018 Idiap Research Institute <http://www.idiap.ch/>
######################################################################

from __future__ import print_function

"""The Person manager performs

- DONE re-identification of the tracklet coming from a person tracker
- DONE fusion between audio and video
- DONE triggers gaze tracking
- TODO vfoa with targetrs and adressee

"""

import os
import sys
import cv2
import time
import copy
import math
import argparse

import rospy
import tf

from datetime import datetime

from cv_bridge import CvBridge

from collections import deque

from std_msgs.msg import Int32
from std_msgs.msg import ColorRGBA

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped

from perception_msgs.msg import FaceInfo
from perception_msgs.msg import GazeInfo
from perception_msgs.msg import BoundingBox
from perception_msgs.msg import BodyJoints3D
from perception_msgs.msg import FaceInfoArray
from perception_msgs.msg import GazeInfoArray
from perception_msgs.msg import TrackedPerson
from perception_msgs.msg import VoiceActivity
from perception_msgs.msg import TrackedPerson
from perception_msgs.msg import SoundSourceArray
from perception_msgs.msg import VoiceActivityArray
from perception_msgs.msg import TrackedPersonArray
from perception_msgs.msg import PersonTrackletArray

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from person import Person

from utils import COLORS
from utils import montage
from utils import draw_bounding_box
from utils import scale_bounding_box
from utils import get_closest_rosmsg
from utils import project_2d_to_3d_area
from utils import draw_facial_landmarks
from utils import project_head_on_depth
from utils import project_2d_body_joints_on_depth
from orgreport import OrgReport

from pytopenface import OpenfaceComputer
from pytopenface import enlarge_bounding_box
from pytopenface import Reidentifier

try:
    from vfoa.vfoa_module import VFOAModule
    from vfoa.vfoa_module import Person as VFOAPerson
    from vfoa.vfoa_module import Target as VFOATarget
    from vfoa.utils.geometry import vectorToYawElevation
    VFOA_MODULE_FOUND = True
except:
    VFOA_MODULE_FOUND = False

import numpy as np

from image_tiler import ImageTiler


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def ros_timestamp_to_str(secs, nsecs):
    t = datetime.fromtimestamp(secs)
    return "{}.{}".format(t.strftime("%Y-%m-%d %H:%M:%S"),
                          nsecs)

def list_to_minmax_tuple(id_list):
    """Return a set of tuples made of all pairwise element of input
       list. The first element of the tuple is smaller than the
       second.

    """
    pairs = set([])
    n_ids = len(id_list)
    for i in range(0, n_ids-1):
        for j in range(1, n_ids):
            mini = min(id_list[i], id_list[j])
            maxi = max(id_list[i], id_list[j])
            pairs.add((mini, maxi))
    return pairs

def crop_bounding_box(image,
                      h, w,
                      height, width,
                      image_height=-1, image_width=-1):
    """Return a crop of the image defined by input bounding box
    parameters

    """
    if image_height < 0:
        image_height = image.shape[0]
    if image_width < 0:
        image_width = image.shape[0]
    W = float(image.shape[1])/float(image_width)
    H = float(image.shape[0])/float(image_height)
    x0 = int(max(w*W, 0))
    y0 = int(max(h*H, 0))
    x1 = int(min((w + width)*W, image.shape[1]-1))
    y1 = int(min((h + height)*H, image.shape[0]-1))
    return image[int(y0):int(y1),int(x0):int(x1)]

class PersonManager(object):
    """
    """
    def __init__(self,
                 color_topic="/color/image_raw",
                 depth_topic="/depth/image_raw",
                 info_topic="/depth/camera_info",
                 tracklet_topic="/tracklet",
                 sound_topic="/audio_perception/ssl",
                 person_of_interest_topic="/person_of_interest",
                 dlib_landmarks_filename="",
                 use_cuda=1,
                 use_gaze_tracker=False,
                 use_vfoa_module=False,
                 max_length_buffer=30,
                 rospy_rate=30,
                 min_votes_for_reid=10,
                 keep_threshold=0.1,
                 reid_threshold=0.4,
                 max_nb_features_per_id=400,
                 angle_tolerance=5,
                 project_on_depth=False,
                 visu=0,
                 status_every=100,
                 record="",
                 report=""):
        """
        """
        rospy.loginfo("keep_threshold {}".format(keep_threshold))
        rospy.loginfo("reid_threshold {}".format(reid_threshold))
        rospy.loginfo("max_nb_features_per_id {}".format(max_nb_features_per_id))

        self.rate = rospy_rate
        self.rospy_rate = rospy.Rate(rospy_rate)
        self.max_length_buffer = max_length_buffer
        self.visu = visu
        self.project_on_depth = project_on_depth

        self.use_gaze_tracker = use_gaze_tracker

        self.record = record
        if len(self.record) > 0 and not os.path.exists(self.record):
            os.mkdir(self.record)

        self.report = None
        if len(report) > 0:
            self.report = OrgReport(report,
                                    title="Person Manager",
                                    options="toc:nil")

        # Print status info every N loops
        self.status_every = status_every

        # Buffers
        self.color_buffer    = deque(maxlen=self.max_length_buffer)
        self.depth_buffer    = deque(maxlen=self.max_length_buffer)
        self.tracklet_buffer = deque(maxlen=self.max_length_buffer)
        # self.sound_buffer    = deque(maxlen=self.max_length_buffer)

        self.cvbridge = CvBridge()

        self.tf_listener = tf.TransformListener()

        # Times to monitor last messages
        self.rostime_last_msg = 0 # Time of ROS msg (rosbag, live, etc.)
        self.abstime_last_msg = 0 # Time of computer

        # Depth camera parameters
        self.depth_fx = 0
        self.depth_fy = 0
        self.depth_cx = 0
        self.depth_cy = 0

        # Persons list
        #
        #  - known_persons: All known persons from the beginning of
        #    time
        #
        #  - front_persons: Persons who are currently interacting with
        #    the robot, they are not necessarily visible, if they are
        #    speaking outside the field of view. This is a subset of
        #    'known_persons': when a person does not interact with the
        #    robot anymore, she is removed from the 'front_persons'
        #    list, but remains in the 'known_persons'.
        #
        self.known_persons = []
        self.front_persons = []
        # self.persons = []

        # The current persons which we focus our attention on
        self.person_of_interest = []

        ############################################################
        # Reidentification part
        ############################################################
        # Minimum number of votes for an identity to:
        #  - re-identify an non-identified tracklet
        #  - merge to tracklets
        self.min_votes_for_reid = min_votes_for_reid

        # Used when reidentification is performed
        self.tracklet_id_to_person_id = {}

        # Lists of tracklets that should not be merged together
        # because obviously being of different persons.
        #
        # For instance, if mutually_exclusive is [[1,2,4], [3,2]]
        # (which would mean that persons 1,2,4 have been seen together
        # on the smae image, and later persons 2,3 also) and that for
        # some reason (e.g. occlusion), the merge process wants to
        # merge [1,2], then this is forbidden.
        self.mutually_exclusive = set([])

        # A dictionnary to accumulate votes for each tracklet. When
        # the number of votes is larger than self.min_votes_for_reid
        self.tracklet_votes = {}

        self.openface_computer = OpenfaceComputer(dlib_landmarks_filename,
                                                  useCuda=use_cuda)

        self.reidentifier = Reidentifier(keep_threshold=keep_threshold,
                                         reid_threshold=reid_threshold,
                                         max_nb_features_per_id=max_nb_features_per_id)


        ############################################################
        # VFOA module
        ############################################################
        vfoa_model = "HMM" # geometricModel, gazeProbability
        self.vfoa_module = VFOAModule(model=vfoa_model,
                                      history_duration=30) if (VFOA_MODULE_FOUND and use_vfoa_module) else None
        # if self.vfoa_module is not None:
        #     self.vfoa_module.model.prob_aversion = 1e-6
        ############################################################

        ############################################################
        # Sound source / tracking matching part
        ############################################################
        self.angle_tolerance = angle_tolerance

        ############################################################

        # self.shape_predictor = None
        # if len(dlib_landmarks_filename) > 0:
        #     rospy.loginfo("Opening {}".format(dlib_landmarks_filename))
        #     self.shape_predictor = dlib.shape_predictor(dlib_landmarks_filename)

        # self.face_detector = dlib.get_frontal_face_detector()

        # Subscribers
        queue_size = 10

        rospy.loginfo("Subscribe to {}".format(color_topic))
        self.color_sub = None
        self.use_compressed_color = False
        if color_topic.endswith("compressed"):
            self.color_sub = rospy.Subscriber(color_topic,
                                              CompressedImage,
                                              self.__color_image_cb,
                                              queue_size=queue_size)
            self.use_compressed_color = True
        else:
            self.color_sub = rospy.Subscriber(color_topic,
                                              Image,
                                              self.__color_image_cb,
                                              queue_size=queue_size)

        rospy.loginfo("Subscribe to {}".format(depth_topic))
        self.depth_sub = rospy.Subscriber(depth_topic,
                                          Image,
                                          self.__depth_image_cb,
                                          queue_size=queue_size)

        rospy.loginfo("Subscribe to {}".format(info_topic))
        self.depth_info_sub = rospy.Subscriber(info_topic,
                                               CameraInfo,
                                               self.__depth_info_cb)

        rospy.loginfo("Subscribe to {}".format(tracklet_topic))
        self.tracklet_sub = rospy.Subscriber(tracklet_topic,
                                             PersonTrackletArray,
                                             self.__tracklet_cb,
                                             queue_size=1)

        rospy.loginfo("Subscribe to {}".format(sound_topic))
        self.sound_sub = rospy.Subscriber(sound_topic,
                                          SoundSourceArray,
                                          self.__sound_cb,
                                          queue_size=10)

        rospy.loginfo("Subscribe to {}".format(person_of_interest_topic))
        self.poi_sub = rospy.Subscriber(person_of_interest_topic,
                                        Int32,
                                        self.__person_of_interest_cb,
                                        queue_size=10)

        ############################################################
        # Publishers
        ############################################################

        self.tracked_person_pub = rospy.Publisher("/wp2/track",
                                                  TrackedPersonArray,
                                                  queue_size = 5)

        self.gaze_info_pub = rospy.Publisher("/wp2/gaze",
                                             GazeInfoArray,
                                             queue_size = 5)

        self.face_info_pub = rospy.Publisher("/wp2/face",
                                             FaceInfoArray,
                                             queue_size = 5)

        self.voice_activity_pub = rospy.Publisher("/wp2/voice",
                                                  VoiceActivityArray,
                                                  queue_size = 5)

        self.head_pose_pub = rospy.Publisher("/head_viz_debug",
                                             MarkerArray,
                                             queue_size = 10)

        self.gaze_trigger_pub = rospy.Publisher("/hg3d/trigger",
                                                BoundingBox,
                                                queue_size = 5)


    def __new_msg(self, header):
        """Set rostime_last_msg and abstime_last_msg and return the time in ns
        of the input message

        """
        # Save the time of the current message (can be in the past if
        # rosbag play is being used
        ns = header.stamp.to_nsec()

        if self.rostime_last_msg == 0 and self.abstime_last_msg == 0:
            self.abstime_last_msg = int(time.time()*1e9)
            self.rostime_last_msg = ns
        else:
            if ns < self.rostime_last_msg:
                self.__clear_buffers()
            self.abstime_last_msg = int(time.time()*1e9)
            self.rostime_last_msg = ns

        return ns


    def __clear_buffers(self):
        """Clear all buffers. Useful when time goes back in past with rosbag
        play -l

        """
        abs_str = time.ctime(self.abstime_last_msg/1e9)
        ros_str = time.ctime(self.rostime_last_msg/1e9)
        print("######### Go back in time {} {}".format(abs_str, ros_str))

        self.color_buffer.clear()
        self.depth_buffer.clear()
        self.tracklet_buffer.clear()
        # self.persons = [] # del self.persons[:]  ?????

    def __person_of_interest_cb(self, imsg):
        """
        """
        poi = imsg.data
        if poi not in self.person_of_interest:
            rospy.loginfo("Setting person of interest to {}".format(poi))
            self.person_of_interest = [poi]

    def __color_image_cb(self, imsg):
        """
        """
        # ns = self.__new_msg(imsg.header)
        # self.color_buffer.append(imsg)
        self.color_buffer.append(copy.deepcopy(imsg))
        # print("color {}".format(ros_timestamp_to_str(imsg.header.stamp.secs,
        #                                              imsg.header.stamp.nsecs)))
        rospy.logdebug("Receive color {}". \
                          format(imsg.header.stamp.to_nsec()))


    def __depth_image_cb(self, imsg):
        """
        """
        # ns = self.__new_msg(imsg.header)
        # self.depth_buffer.append(imsg)
        self.depth_buffer.append(copy.deepcopy(imsg))
        rospy.logdebug("Receive depth {}". \
                          format(imsg.header.stamp.to_nsec()))


    def __depth_info_cb(self, imsg):
        """
        """
        self.depth_fx, _, self.depth_cx, \
            _, self.depth_fy, self.depth_cy, _, _, _ = imsg.K


    def __tracklet_cb(self, imsg):
        """
        """
        self.tracklet_buffer.append(imsg)

        tic = time.time()

        rostime = imsg.header.stamp.to_nsec()

        # Get the image streams
        cmsg, diff = get_closest_rosmsg(rostime, self.color_buffer)

        if cmsg is None:
            rospy.loginfo("No image found for {}".format(rostime))
            return

        if self.use_compressed_color:
            color = self.cvbridge.compressed_imgmsg_to_cv2(cmsg, "bgr8").copy()
        else:
            color = self.cvbridge.imgmsg_to_cv2(cmsg, "bgr8").copy()

        depth = None
        dmsg, diff = get_closest_rosmsg(rostime, self.depth_buffer)
        if dmsg is not None:
            depth = self.cvbridge.imgmsg_to_cv2(dmsg, "passthrough")


        # List IDs that are present at the same time
        visible_ids = set([])
        present_ids = [t.tracklet_id for t in imsg.data]

        # Create a new person if not already present
        for tracklet_msg in imsg.data:
            ID = tracklet_msg.tracklet_id
            if ID not in self.tracklet_id_to_person_id:
                rospy.loginfo("Create person {}".format(ID))
                self.tracklet_id_to_person_id[ID] = ID

        ######################################################################
        # DEBUG
        ######################################################################
        # Simply fill front_persons with the tracklets, no
        # reidentification is performed
        ######################################################################
        # self.front_persons = []

        # for tracklet_msg in imsg.data:
        #     tid = tracklet_msg.tracklet_id
        #     pid = self.tracklet_id_to_person_id[tid]

        #     p = self.get_person_with_id(pid)

        #     if p is None:
        #         p = Person(self.max_length_buffer)
        #         self.known_persons.append(p)
        #     else:
        #         print("Got {}".format(pid))

        #     p.update(tracklet_msg)
        #     p.person_id = pid
        #     self.front_persons.append(p)

        # print("Known {} Front {}".format(len(self.known_persons),
        #                                  len(self.front_persons)))
        ######################################################################

        # Make pair of IDs that are mutually exclusive to avoid
        # merging 2 of them
        n_ids = len(present_ids)
        for i in range(0, n_ids-1):
            for j in range(i+1, n_ids):
                # Tracklet ID and person ID for first element i
                tidi = present_ids[i]
                pidi = self.tracklet_id_to_person_id[tidi]
                # Tracklet ID and person ID for first element j
                tidj = present_ids[j]
                pidj = self.tracklet_id_to_person_id[tidj]

                self.add_mutually_exclusive(tidi, tidj)
                self.add_mutually_exclusive(pidi, tidj)
                self.add_mutually_exclusive(tidi, pidj)
                self.add_mutually_exclusive(pidi, pidj)

        # if self.reidentifier.features is not None:
        #     print("gallery {}".format(self.reidentifier.features.shape))

        # print("Votes {}".format(self.tracklet_votes))

        # Vote for ids
        lists_to_merge = []
        for tracklet_msg in imsg.data:
            bb = tracklet_msg.box
            x0, x1, y0, y1 = scale_bounding_box(color, bb.h, bb.w,
                                                bb.height, bb.width,
                                                bb.image_height, bb.image_width)
            x2, x3, y2, y3 = enlarge_bounding_box(x0, x1, y0, y1, color, 0.1)

            # Extract face (no copy)
            crop = color[y2:y3,x2:x3]

            features = self.openface_computer.compute_on_image(crop, 1)
            # features, debug = self.openface_computer.compute_on_image(crop, 1)

            tid = tracklet_msg.tracklet_id
            pid = self.tracklet_id_to_person_id[tid]

            if features.face is not None and features.visu is not None:
                trombi = montage([crop, features.visu, features.face])
                # cv2.imshow("Trombi", trombi)
                # cv2.waitKey(1)
                if self.report is not None:
                    sub_dir = "{}".format(rostime)
                    image_name = "{}_pid_{}_tid_{}.jpg".format(rostime, pid, tid)
                    self.report.add_image(trombi, image_name, sub_dir)

            # rospy.loginfo("Tracklet {} as {}".format(tid, pid))

            if features.features is not None:
                # self.reidentifier.add(features.features, pid, crop)
                self.reidentifier.add(features.features, pid)
                reid = self.reidentifier.identify(features.features)
                self.accumulate_votes(reid.votes, pid, tid)

            # Perform merging of identities
            for tracklet_id, votes_per_id in self.tracklet_votes.items():
                ids_to_merge = set([tracklet_id])
                for other_id, n_votes in votes_per_id.items():
                    if n_votes >= self.min_votes_for_reid:
                        ids_to_merge.add(other_id)
                if len(ids_to_merge) > 1:
                    lists_to_merge.append(list(ids_to_merge))

            # Get freshly updated person ID
            # pid = self.tracklet_id_to_person_id[tid]
            p = self.get_person_with_id(pid)
            # print("PERSON {}/{}".format(pid,tid))

            if p is None:
                p = Person(self.max_length_buffer)
                self.known_persons.append(p)

            if features.features is not None:
                p.is_identified += 1

            joints3d = BodyJoints3D()

            # Deep copy
            head = Pose()
            head.position.x = tracklet_msg.head.position.x
            head.position.y = tracklet_msg.head.position.y
            head.position.z = tracklet_msg.head.position.z
            head.orientation.x = tracklet_msg.head.orientation.x
            head.orientation.y = tracklet_msg.head.orientation.y
            head.orientation.z = tracklet_msg.head.orientation.z
            head.orientation.w = tracklet_msg.head.orientation.w

            if depth is not None:
                joints3d = project_2d_body_joints_on_depth(color, depth,
                                                           tracklet_msg,
                                                           self.depth_fx,
                                                           self.depth_fy,
                                                           self.depth_cx,
                                                           self.depth_cy,
                                                           5)
                if self.project_on_depth:
                    head3d = project_head_on_depth(depth,
                                                   tracklet_msg,
                                                   self.depth_fx,
                                                   self.depth_fy,
                                                   self.depth_cx,
                                                   self.depth_cy,
                                                   10)

                    # When the head is on the border of the color image,
                    # depth info may not be available, which result in
                    # (0,0,0) location. In this case, we use the
                    # estimation based on height of face.
                    if head3d.position != Point(0,0,0):
                        head = head3d

            # p.update(tracklet_msg)
            p.add_tracklet_msg(rostime, tracklet_msg)
            p.add_joints_msg(rostime, joints3d)
            p.add_head_pose_msg(rostime, head)
            p.person_id = pid
            p.track_id  = tid


        # Prepare IDs to merge.
        old_to_new = {}
        for list_to_merge in lists_to_merge:
            new_id = min(list_to_merge)
            for theid in list_to_merge:
                if theid not in old_to_new:
                    old_to_new[theid] = new_id
                else:
                    old_to_new[theid] = min(new_id, old_to_new[theid])

        # Recursively update the 'old_to_new'. If 5 should be
        # reidentified as 4, and 4 as 3, then 5 should be reidentified
        # as 3.
        n_changes = 1
        while n_changes > 0:
            n_changes = 0
            for old, new in old_to_new.items():
                if old_to_new[new] != new:
                    n_changes += 1
                    old_to_new[old] = old_to_new[new]

        ids_to_merge = {}
        for old, new in old_to_new.items():
            if new not in ids_to_merge:
                ids_to_merge[new] = set([])
            ids_to_merge[new].add(old)

        for new_id in ids_to_merge:
            self.merge_persons(new_id, list(ids_to_merge[new_id]))

        # When all persons are reidentified, we update
        # 'front_persons'. We don't do it at the same time of
        # reidentification because this callback runs in parallel of
        # the 'run' main loop and it can cause some access
        # conflicts. And reidentification is time consuming, so by the
        # time the 'run' accesses the 'front_persons', it might be
        # emptied by the callback.
        self.front_persons = []
        for tracklet_msg in imsg.data:
            tid = tracklet_msg.tracklet_id
            pid = self.tracklet_id_to_person_id[tid]

            p = self.get_person_with_id(pid)
            if p is None:
                rospy.logwarn("Cound not find person {}".format(pid))
            else:
                self.front_persons.append(p)

        # print("before vfoa len {}: {}".format(len(self.front_persons), [p.person_id for p in self.front_persons]))

        for p in self.front_persons:
            p.update_identities(self.tracklet_id_to_person_id)

        # toc = time.time()
        # print("Reid callback took {}".format(toc-tic))
        # tic = time.time()

        qcam = tf.transformations.quaternion_from_euler(
            np.radians(90), np.radians(90), np.radians(180))
        qcam = tf.transformations.quaternion_conjugate(qcam)

        personDict = {}

        targetDict = {
            "robot": VFOATarget("robot", np.array([0.0, 0.0, 0.0])),
            # "tablet": VFOATarget("tablet", np.array([0.0, 0.5, 0.0])),
        }

        # toc = time.time()
        # print("Quaternion callback took {}".format(toc-tic))
        # tic = time.time()

        for p in self.front_persons:
            pos3d = p.get_3d_position()
            targetDict[p.person_id] = VFOATarget(p.person_id,
                                                 [pos3d.position.x,
                                                  pos3d.position.y,
                                                  pos3d.position.z],
                                                 positionCS="OCS")

            # Get roll pitch yaw in head coordinate (regular roll, pitch, yaw)
            q = np.array([pos3d.orientation.x, pos3d.orientation.y,
                          pos3d.orientation.z, pos3d.orientation.w])
            # print("[manager] Input quaternion {}".format(q))
            q = tf.transformations.quaternion_multiply(qcam, q)

            euler = tf.transformations.euler_from_quaternion(q)
            roll, pitch, yaw = euler

            yaw = np.degrees(yaw)
            pitch = np.degrees(pitch)
            roll = np.degrees(roll) # Not used, always 0

            # print("[manager] ID {} \t R {:+.2f} \t P {:+.2f} \t Y {:+.2f}"
            #       .format(p.person_id, roll, pitch, yaw))

            headpose = [pos3d.position.x,
                        pos3d.position.y,
                        pos3d.position.z,
                        yaw, pitch, roll]

            # print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(*headpose))
            # gazepose = headpose
            # mid_line_effect = 2
            # gazepose = np.array([pos3d.position.x,
            #                      pos3d.position.y,
            #                      pos3d.position.z,
            #                      mid_line_effect*yaw,
            #                      mid_line_effect*pitch,
            #                      mid_line_effect*roll])

            personDict[p.person_id] = VFOAPerson(p.person_id,
                                                 headpose=headpose,
                                                 positionCS="OCS",
                                                 poseCS="FCS",
                                                 poseUnit="deg")

        # print("personDict.keys() {}".format(personDict.keys()))
        # print("targetDict.keys() {}".format(targetDict.keys()))

        self.vfoa_module.compute_vfoa(personDict, targetDict, rostime/1e9)

        vfoa = self.vfoa_module.get_vfoa()

        for p in self.front_persons:
            pid = int(p.person_id)
            assert pid in vfoa # FIXME that
            p.add_vfoa(rostime, {target: proba for target, proba in vfoa[pid].items()})

        toc = time.time()

        # print("VFOA part of callback took {}".format(toc-tic))


    def __sound_cb(self, imsg):
        """
        """
        rostime = imsg.header.stamp.to_nsec()

        rospy.logdebug("Receive sound {}". \
                          format(imsg.header.stamp.to_nsec()))

        # self.sound_buffer.append(imsg)

        # print("SOUND {} persons in front {} speaking".format \
        #       (len(self.front_persons), len(imsg.data)))

        sound_directions = []
        for m in imsg.data:
            sound_directions.append((m.direction.x, m.direction.y))

        # print("Sound directions {}".format(sound_directions))

        # (trans, rot) = self.tf_listener.lookupTransform("CameraTop_optical_frame",
        #                                                 "Head", rospy.Time(0))
        # print(trans)
        # print(rot)


        tolerance_rad = np.radians(self.angle_tolerance)


        for p in self.front_persons:
            l = p.get_3d_position()
            # TO BE IMPROVED: so far, manual conversion of frame
            # CameraTop_optical_frame and Head
            x = +l.position.z
            y = -l.position.x
            norm = math.sqrt(x*x + y*y)
            if norm < 1e-9: continue
            x = x/norm
            y = y/norm

            is_speaking = 0
            for i, src in enumerate(sound_directions):
                angle = angle_between(np.array(src), np.array((x,y)))
                if angle < tolerance_rad:
                    is_speaking = 1

            if is_speaking > 0:
                p.set_is_speaking(rostime, 1)
            else:
                p.set_is_speaking(rostime, 0)

            # print("i {} p {} angle {} tol {}".format(i, p.person_id, angle, tolerance_rad))

        # print("Person direction {}".format(person_directions))


    def merge_persons(self, new_id, ids_to_merge):
        """Merge 'ids_to_merge' into 'new_id'

           This is a convenient function to call to avoid changing
           directly the __tracklet_cb() part where the merging is
           actually done (if later we add reidentifier based on
           clothes, etc.)

        """
        rospy.loginfo("Merging identities {} into {}". \
                      format(ids_to_merge, new_id))

        # Update correspondance between tracklet id and person id
        for theid in ids_to_merge:
            self.tracklet_id_to_person_id[theid] = new_id

        for other in ids_to_merge:
            if other in self.person_of_interest:
                self.person_of_interest = [new_id]
                rospy.loginfo("Person of interest is now {}".format(new_id))

        # Propagate ids to deleted tracklet.
        #
        # For instance, if person 28 exists, and if tracklet 30 was
        # reidentified as 29, then we have
        #
        #   tracklet_id_to_person_id = {28: 28, 29: 29, 30: 29}
        #
        # If now we identify 29 as being 28, then we also have to
        # propagate 30 to 28, and get
        #
        #   tracklet_id_to_person_id = {28: 28, 29: 28, 30: 28}
        #
        for tid, pid in self.tracklet_id_to_person_id.items():
            if pid in ids_to_merge:
                self.tracklet_id_to_person_id[tid] = new_id

        # print("tracklet_id_to_person_id {}".format(self.tracklet_id_to_person_id))

        for id_to_remove in ids_to_merge:
            if id_to_remove == new_id:
                # Remove counts in favour of deleted id
                # 19: {19:200, 23:30, 24: 12}
                for i in ids_to_merge:
                    if (i != new_id) and (i in self.tracklet_votes[new_id]):
                        del self.tracklet_votes[new_id][i]
            else:
                if id_to_remove in self.tracklet_votes:
                    del self.tracklet_votes[id_to_remove]

        # print("Tracklet votes are now {}".format(self.tracklet_votes))

        self.reidentifier.merge(new_id, ids_to_merge)

        for id_to_remove in ids_to_merge:
            # Target person not merged with herself, it will be kept
            # in the 'known_persons'.
            if new_id == id_to_remove: continue

            source_person = self.get_person_with_id(id_to_remove)
            target_person = self.get_person_with_id(new_id)

            if target_person is None:
                rospy.loginfo("Cannot remove {}, probably already removed". \
                              format(id_to_remove))
            else:
                target_person.merge(source_person)
                self.known_persons.remove(source_person)


    def are_mutually_exclusive(self, id1, id2):
        """Return if (id1,id2) is in self.mutualy_exclusive"""
        if (min(id1,id2), max(id1,id2)) in self.mutually_exclusive:
            return True
        else:
            return False


    def add_mutually_exclusive(self, id1, id2):
        """Add a new tuple to 'mutually_exclusive'"""
        self.mutually_exclusive.add((min(id1,id2), max(id1,id2)))


    def get_person_with_id(self, ID):
        """Return the person with input ID, and return None if does not exist.

        """
        for p in self.known_persons:
            if p.person_id == ID:
                return p
        return None


    def visualise(self, time_ns):
        """Visualise the status of the perception at timem time_ns"""

        rospy.logdebug("Visu time {}".format(time_ns))

        msg, diff = get_closest_rosmsg(time_ns, self.color_buffer)
        image = self.cvbridge.imgmsg_to_cv2(msg, "bgr8")
        display = image.copy()

        for p in self.known_persons:
            # Get position in image
            h, w, height, width, image_height, image_width = p.get_2d_position()
            x0, x1, y0, y1 = scale_bounding_box(display, h, w, height, width,
                                                image_height, image_width)

            pid = p.person_id
            tid = p.tracklet_id
            color = COLORS[1 + pid % (len(COLORS)-1)]
            draw_bounding_box(display, x0, y0, x1, y1,
                              color=color, ID=pid, name=str(tid))

        cv2.imshow("Person Manager", display)
        cv2.waitKey(1)

        if len(self.record) > 0:
            image_name = "{}.jpg".format(time_ns)
            name = os.path.join(self.record, image_name)
            cv2.imwrite(name, display)


    def accumulate_votes(self, votes, person_id, tracklet_id):
        """Accumulate votes to 'tracklet_votes' and take into account the
           'mutually_exclusive' pairs

        """
        # Create if needed
        if person_id not in self.tracklet_votes:
            self.tracklet_votes[person_id] = {}

        accumulated = self.tracklet_votes[person_id]

        for k,v in votes.items():
            p1 = (min(tracklet_id, k), max(tracklet_id, k))
            p2 = (min(person_id,   k), max(person_id,   k))

            # Don't accumulate when mutually exclusive
            if (p1 not in self.mutually_exclusive) and \
               (p2 not in self.mutually_exclusive):
                if k not in accumulated:
                    accumulated[k] = 0
                accumulated[k] += v


    def run(self):
        """
        """
        n_iterations = 0
        was_cleaned = False

        # Remember last published timestamp not to publish several
        # times the same messages
        last_published_timestamp = 0

        previous_person_of_interest = -1

        while not rospy.is_shutdown():
            n_iterations += 1
            # if self.reidentifier.features is not None:
            #     print("VOTES {} GALLERY {}".format(self.tracklet_votes,
            #                                        self.reidentifier.features.shape))

            if len(self.color_buffer) > 0 and len(self.tracklet_buffer) > 0:
                last = list(self.tracklet_buffer)[-1].header.stamp.to_nsec()
                msg, diff = get_closest_rosmsg(last, self.color_buffer)

                # omsg = VoiceActivityArray()
                # omsg.header = imsg.header
                # for m in imsg.data:
                #     print(m.direction)

                # trigger = Point(0, 0, 0)

                if self.use_gaze_tracker:
                    if len(self.person_of_interest) == 0:
                        # Stop the tracker
                        self.gaze_trigger_pub.publish(BoundingBox(0, 0, 0, 0, 0, 0))
                    else:
                        poi = self.person_of_interest[0]

                        if previous_person_of_interest != poi:
                            # Stop the tracker this time
                            self.gaze_trigger_pub.publish(BoundingBox(0, 0, 0, 0, -1, -1))
                            # Tracking will start next frame
                            previous_person_of_interest = poi
                            rospy.loginfo("Person of interest is now {}".format(poi))

                        else:
                            poi_found = False
                            for p in self.front_persons:
                                if p.person_id == poi:
                                    h, w, H, W, Hi, Wi = p.get_2d_position()
                                    bb = BoundingBox(h, w, H, W, Hi, Wi)
                                    pos3d = p.get_3d_position()
                                    self.gaze_trigger_pub.publish(bb)
                                    previous_person_of_interest = poi
                                    poi_found = True

                            if not poi_found:
                                # Stop the tracker
                                self.gaze_trigger_pub.publish(BoundingBox(0, 0, 0, 0, 0, 0))

                # for p in self.front_persons:
                #     if p.person_id in self.person_of_interest:
                #         h, w, H, W, Hi, Wi = p.get_2d_position()
                #         bb = BoundingBox(h, w, H, W, Hi, Wi)
                #         pos3d = p.get_3d_position()
                #         self.gaze_trigger_pub.publish(bb)
                #         previous_person_of_interest = p.person_id

                        # if pos3d.position.z < 1.2:
                        #     self.gaze_trigger_pub.publish(bb)
                        # else:
                        #     self.gaze_trigger_pub.publish(BoundingBox(0, 0, 0, 0, 0, 0))

                # ######################################################################
                # # print("="*70)
                # targets = {
                #     "robot": VFOATarget("robot", np.array([0, 0, 0])),
                #     "pitch": VFOATarget("pitch", np.array([0, 1, 0])),
                # }
                # persons = {}
                # for p in self.front_persons:
                #     pos3d = p.get_3d_position()

                #     # print("[manager] Input location x {} y {} z {}".format(pos3d.position.x,
                #     #                                                        pos3d.position.y,
                #     #                                                        pos3d.position.z))
                #     targets[p.person_id] = VFOATarget(p.person_id,
                #                                       np.array([pos3d.position.x,
                #                                                 -pos3d.position.y,
                #                                                 -pos3d.position.z]))

                #     # Get rol pitch yaw in head coordinate (regular roll, pitch, yaw)
                #     qcam = tf.transformations.quaternion_from_euler(np.radians(90),
                #                                                     np.radians(90),
                #                                                     np.radians(180))
                #     qcam = tf.transformations.quaternion_conjugate(qcam)
                #     q = np.array([pos3d.orientation.x, pos3d.orientation.y,
                #                   pos3d.orientation.z, pos3d.orientation.w])
                #     # print("[manager] Input quaternion {}".format(q))
                #     q = tf.transformations.quaternion_multiply(qcam, q)

                #     euler = tf.transformations.euler_from_quaternion(q)
                #     roll, pitch, yaw = euler

                #     yaw = np.degrees(-yaw)
                #     pitch = np.degrees(pitch)
                #     roll = np.degrees(roll) # Not used, always 0

                #     print("[manager] ID {} \t R {:+.2f} \t P {:+.2f} \t Y {:+.2f}"
                #           .format(p.person_id, roll, pitch, yaw))

                #     headpose = np.array([pos3d.position.x,
                #                          -pos3d.position.y,
                #                          -pos3d.position.z,
                #                          yaw, pitch, roll])
                #     mid_line_effect = 2
                #     gazepose = np.array([pos3d.position.x,
                #                          -pos3d.position.y,
                #                          -pos3d.position.z,
                #                          mid_line_effect*yaw,
                #                          mid_line_effect*pitch,
                #                          mid_line_effect*roll])

                #     persons[p.person_id] = VFOAPerson(p.person_id, headpose, gazepose)

                # # for p in persons:
                # #     persons[p].print_person()
                # # for t in targets:
                # #     targets[t].print_target()

                # # print(targets)
                # # print(persons)
                # if self.vfoa_module is not None:
                #     # print(targets)
                #     # print(persons)
                #     self.vfoa_module.compute_vfoa(persons, targets, 1)
                #     vfoa = self.vfoa_module.get_vfoa()
                #     # print(vfoa)
                #     for p in vfoa:
                #         best_target = None
                #         best_score = 0
                #         for i, (target, score) in enumerate(vfoa[p].items()):
                #             if i == 0 or score > best_score:
                #                 best_score = score
                #                 best_target = target
                #         print("{} is looking at {} ({})".format(p, best_target, best_score))
                # ######################################################################

                if last != last_published_timestamp:
                    # Publish messages
                    tracked_person_msg = TrackedPersonArray()
                    gaze_info_msg      = GazeInfoArray()
                    face_info_msg      = FaceInfoArray()
                    voice_activity_msg = VoiceActivityArray()

                    # Header equal to color header
                    tracked_person_msg.header = msg.header
                    gaze_info_msg.header      = msg.header
                    face_info_msg.header      = msg.header
                    voice_activity_msg.header = msg.header

                    timestamp = msg.header.stamp.to_nsec()

                    for p in self.front_persons:
                        tracked_person_msg.data.append(
                            p.to_tracked_person_msg(timestamp))
                        face_info_msg.data.append(
                            p.to_face_info_msg(timestamp))
                        gaze_info_msg.data.append(
                            p.to_gaze_info_msg(timestamp))
                        voice_activity_msg.data.append(
                            p.to_voice_activity_msg(timestamp))

                    self.tracked_person_pub.publish(tracked_person_msg)
                    self.gaze_info_pub.publish(gaze_info_msg)
                    self.face_info_pub.publish(face_info_msg)
                    self.voice_activity_pub.publish(voice_activity_msg)

                    # print("publish {}".format(tracked_person_msg.header.stamp))
                    # print(tracked_person_msg.data[0].body_joints_2d.nose)

                    # print("publish {}".format(last))
                    last_published_timestamp = last
                # else:
                #     print("{} already published {}". \
                #           format(last, last_published_timestamp))

                if len(self.front_persons) > 0:
                    was_cleaned = False

                if len(self.front_persons) == 0 and (n_iterations % self.status_every) == 0:
                    # rospy.loginfo("The gallery contains {} elements"
                    #               .format(len(self.reidentifier.ids)))
                    if not was_cleaned:
                        self.reidentifier.print_status()
                        self.reidentifier.clean()
                        was_cleaned = True

                # if self.report is not None:
                #     self.report.add_text("\n* {}\n".format(last))
                #     available_ids = self.reidentifier.get_available_ids()
                #     self.report.add_text("Available ids {}\n\n".format(available_ids))
                #     for ID in available_ids:
                #         images = self.reidentifier.get_images_for_id(ID)
                #         tiler = ImageTiler()
                #         tiler.add_images(images)
                #         image_name = "{}_id_{}.jpg".format(last, ID)
                #         sub_dir = "{}".format(last)
                #         self.report.add_image(tiler.montage, image_name, sub_dir)


            # self.rospy_rate.sleep()
            time.sleep(1.0/self.rate)

if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--color",
                        type=str,
                        default="/naoqi_driver_node/camera/front/image_raw",
                        help="Registered color topic")
    parser.add_argument("--depth",
                        type=str,
                        default="/naoqi_driver_node/camera/depth/image_raw",
                        help="Registered depth topic")
    parser.add_argument("--info",
                        type=str,
                        default="/naoqi_driver_node/camera/depth/camera_info",
                        help="Registered depth topic")
    parser.add_argument("--tracklet",
                        type=str,
                        default="/tracklet",
                        help="Topic name publishing " \
                             "perception_msgs/PersonTracklet")
    parser.add_argument("--sound",
                        type=str,
                        default="/audio_perception/ssl",
                        help="Topic name publishing " \
                             "perception_msgs/SoundSourceArray.msg")
    parser.add_argument("--buffer-length",
                        type=int,
                        default=120,
                        help="Number of message to keep in memory")
    parser.add_argument("--use-cuda-if-available",
                        type=int,
                        default=1,
                        help="Whether to run the reidentification on GPU")
    parser.add_argument("--dlib",
                        type=str,
                        default="",
                        help="Path to shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--keep-threshold",
                        type=float,
                        default=0.25,
                        help="Threshold below which images are not " \
                             "accumulated")
    parser.add_argument("--reid-threshold",
                        type=float,
                        default=0.40,
                        help="Threshold below features vote for " \
                             "a given identity")
    parser.add_argument("--max-nb-features-per-id",
                        type=int,
                        default=200,
                        help="Nb of features to store per identity")
    parser.add_argument("--hz",
                        type=int,
                        default=10,
                        help="How fast to call main loop (n hz)")
    parser.add_argument("--use-gaze-tracker",
                        type=bool,
                        default=False,
                        help="Whether HG3D is launched")
    parser.add_argument("--use-vfoa-module",
                        type=bool,
                        default=False,
                        help="Whether to use external VFOA module")
    parser.add_argument("--visu",
                        type=int,
                        default=0,
                        help="Level of visualisation")
    parser.add_argument('--project-on-depth',
                        action='store_true',
                        help="Whether to use depth for localising in 3D")
    parser.add_argument("--record",
                        type=str,
                        default="",
                        help="Directory where to save visualisation")
    parser.add_argument("--report",
                        type=str,
                        default="",
                        help="Directory where to save a report of what happened")

    try:
        opts = parser.parse_args(rospy.myargv()[1:])
    except:
        parser.print_help()
        sys.exit(1)

    try:
        rospy.init_node("person_manager_node2")
        rospy.loginfo("Starting person_manager_node")
        pm=PersonManager(color_topic=opts.color,
                         depth_topic=opts.depth,
                         info_topic=opts.info,
                         tracklet_topic=opts.tracklet,
                         sound_topic=opts.sound,
                         max_length_buffer=opts.buffer_length,
                         rospy_rate=opts.hz,
                         dlib_landmarks_filename=opts.dlib,
                         keep_threshold=opts.keep_threshold,
                         reid_threshold=opts.reid_threshold,
                         max_nb_features_per_id=opts.max_nb_features_per_id,
                         use_cuda=opts.use_cuda_if_available,
                         use_gaze_tracker=opts.use_gaze_tracker,
                         use_vfoa_module=opts.use_vfoa_module,
                         project_on_depth=opts.project_on_depth,
                         visu=opts.visu,
                         record=opts.record,
                         report=opts.report)
        pm.run()
    except rospy.ROSInterruptException:
        pass
