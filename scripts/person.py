######################################################################
# Copyright (c) 2018 Idiap Research Institute <http://www.idiap.ch/>
######################################################################

from __future__ import print_function

import os
import sys
import math
import copy
import logging

import numpy as np

from collections import deque

import rospy

logger = logging.getLogger(__name__)

try:
    from perception_msgs.msg import FaceInfo
    from perception_msgs.msg import GazeInfo
    from perception_msgs.msg import TrackedPerson
    from perception_msgs.msg import VoiceActivity
    from perception_msgs.msg import IdWithProbability
except:
    logger.error("Cannot import perception_msgs")

from utils import StampedData

class Person(object):
    """A Person with a history for its attributes

    Args:
        max_length_buffer:
        silence_turn_secs: Length of silence to end a turn (seconds)

    """
    def __init__(self,
                 max_length_buffer=120,
                 silence_turn_secs=1.0):
        """

        """
        self.max_length_buffer = max_length_buffer

        self.person_id = -1
        self.track_id  = -1

        # self.is_speaking = 0
        self.time_last_is_speaking = -1
        self.time_start_turn = -1
        # self.time_last_turn = -1
        # Buffer of (timestamp_ns, is_speaking)
        self.is_speaking_history = deque(maxlen=10*self.max_length_buffer)

        # Buffer of (timestamp_ns, PersonTracklet)
        self.tracklet_buffer = deque(maxlen=self.max_length_buffer)

        # Buffer of (timestamp_ns, BodyJoints3D)
        self.joints_buffer = deque(maxlen=self.max_length_buffer)

        # Buffer of (timestamp_ns, geometry_msgs/pose)
        self.head_pose_buffer = deque(maxlen=self.max_length_buffer)

        # Buffer of (timestamp_ns, vfoa) when using external VFOA module
        self.vfoa_buffer = StampedData(maxlen=1000)

        self.silence_turn_secs = silence_turn_secs

        self.position_3d = None  # Last known position (x, y, z)
        self.joints_3d   = None  # perception_msgs/BodyJoints3D

        self.alternate_ids = []

        # Indicate whether re-idetification features are being
        # accumulated for this person
        self.is_identified = 0


    # def update(self, imsg):
    #     """Update state of person from a PersonTracklet message, usually
    #     element of PersonTrackletArray.

    #     """
    #     self.person_id = imsg.tracklet_id
    #     self.track_id  = imsg.tracklet_id
    #     self.tracklet_buffer.append(imsg)

    def update_identities(self, identities):
        """When reid is done, update old:new identities"""
        for old, new in identities.items():
            self.vfoa_buffer.update_key(old, new)

    def add_tracklet_msg(self, timestamp_ns, msg):
        """Add (timestamp_ns, msg) to tracklet_buffer."""
        self.tracklet_buffer.append((timestamp_ns, msg))


    def add_joints_msg(self, timestamp_ns, msg):
        """Add (timestamp_ns, msg) to joints_buffer."""
        self.joints_buffer.append((timestamp_ns, msg))


    def add_head_pose_msg(self, timestamp_ns, msg):
        """Add (timestamp_ns, msg) to head_pose_buffer."""
        self.head_pose_buffer.append((timestamp_ns, msg))

    def set_is_speaking(self, now_ns, is_speaking):
        """Set the raw data of whether the person is actually speaking

        Args:
            now_ns: Timestamp now in nanoseconds

        """
        was_speaking_before = self.is_speaking()
        # print("was_speaking_before {}".format(was_speaking_before))
        if is_speaking:
            self.time_last_is_speaking = now_ns

        self.is_speaking_history.append((now_ns, is_speaking))

        is_speaking_now = self.is_speaking()
        # print("is_speaking_now {}".format(is_speaking_now))

        if not was_speaking_before and is_speaking_now:
            self.time_start_turn = now_ns

        if was_speaking_before and not is_speaking_now:
            self.time_start_turn = -1

        # print("self.time_start_turn {}".format(self.time_start_turn))


    def get_turn_duration(self, now_ns=-1):
        """
        """
        turn_duration = 0
        if now_ns < 0:
            if len(self.is_speaking_history):
                now_ns = self.is_speaking_history[-1][0]
            else:
                return turn_duration
        if self.is_speaking():
            turn_duration = now_ns - self.time_start_turn
        return turn_duration

    def is_speaking(self, timestamp_ns=-1):
        """A person is speaking if it has spoken once between now and
        self.silence_turn_secs ago.

        """
        # hist = [is_speaking for _, is_speaking in self.is_speaking_history]
        # nb_times_speaking_lately = np.sum(hist)
        # is_speaking = 1 if nb_times_speaking_lately > 0 else 0
        # print("{} {} {}".format(self.person_id, hist, is_speaking))
        is_speaking = False

        if timestamp_ns < 0:
            if len(self.is_speaking_history):
                timestamp_ns = self.is_speaking_history[-1][0]

        if timestamp_ns < 0: # No data
            return is_speaking

        # Entries should be in temporal order (timestamp, is_speaking)

        # Avoid "RuntimeError: deque mutated during iteration"
        is_speaking_history = copy.deepcopy(self.is_speaking_history)

        for ts, speaking in reversed(is_speaking_history):
            diff = timestamp_ns - ts
            if diff < 0:
                continue
            if diff > self.silence_turn_secs*1e9:
                break
            if speaking:
                is_speaking = True
                break

        return is_speaking

    def get_2d_position(self):
        """Return the coordinates of the bounding box in 2D image."""
        h = 0 # Starting point
        w = 0 # Starting point
        height = 0
        width = 0
        image_height = 0
        image_width = 0
        if len(self.tracklet_buffer) > 0:
            box = self.tracklet_buffer[-1][1].box
            h = box.h
            w = box.w
            height = box.height
            width = box.width
            image_width = box.image_width
            image_height = box.image_height

        return h, w, height, width, image_height, image_width


    def add_vfoa(self, timestamp_ns, vfoa):
        """Add a new element of VFOA

        Args:
            timestamp_ns: Time in ns
            vfoa: Dictionnary of {target_name: proba}

        """
        self.vfoa_buffer.add(timestamp_ns, vfoa)

    def get_3d_position(self):
        """Return the location of the person in 3D."""
        # return self.position_3d[0], self.position_3d[1], self.position_3d[2]
        if len(self.head_pose_buffer) > 0:
            return self.head_pose_buffer[-1][1]
        else:
            return None


    def merge(self, other):
        """Accumulate the messages from 'other' person into current (self)
           person.

        Args:
            other : The other person to merge into self

        """
        self.alternate_ids.append(other.person_id)

        self.is_identified += other.is_identified

        # print("="*70)
        # print(self.vfoa_buffer)
        # print("-"*70)
        # print(other.vfoa_buffer)
        # print("="*70)

        self.vfoa_buffer.merge(other.vfoa_buffer)

        # print(self.vfoa_buffer)
        # print("="*70)

        # TODO: Here we assume that all messages form 'other' person
        # are more recent than the current person. So we replace all
        # the messages from 'self' by the ones from 'other'.
        self.tracklet_buffer.clear()
        for msg in other.tracklet_buffer:
            self.tracklet_buffer.append(msg)


    def get_message_at(self, in_buffer, timestamp=None):
        """Return the message at specified 'timestamp'.

           If 'timestamp', return the last one available.

        """
        tmsg = None

        if len(in_buffer) > 0:
            tmsg = in_buffer[-1]

            # if timestamp is not None:
            #     found = 0
            #     for t in reversed(self.tracklet_buffer):
            #         if timestamp == t[0]:
            #             tmsg = t[1]
            #             found = 1
            #             break
            #     if found == 0:
            #         print("Could not find {}".format(timestamp))

        return tmsg


    def to_tracked_person_msg(self, timestamp=None):
        """Return a TrackedPerson message at time 'timestamp'.

        If 'timestamp' is None, return the lastest available.

        """
        msg = TrackedPerson()
        msg.person_id     = self.person_id
        msg.track_id      = self.track_id
        msg.alternate_ids = self.alternate_ids
        msg.is_identified = self.is_identified

        tmsg = self.get_message_at(self.tracklet_buffer, timestamp)

        if tmsg is not None:
            tmsg = tmsg[1]
            msg.body_joints_2d = tmsg.joints
            msg.box = tmsg.box

            jmsg = self.get_message_at(self.joints_buffer, timestamp)
            if jmsg is not None:
                msg.body_joints_3d = jmsg[1]
            else:
                rospy.logwarn("Cannot find joint message")

            hmsg = self.get_message_at(self.head_pose_buffer, timestamp)
            if hmsg is not None:
                msg.head_pose = hmsg[1]
                x = msg.head_pose.position.x
                y = msg.head_pose.position.y
                z = msg.head_pose.position.z
                msg.head_distance = math.sqrt(x*x + y*y + z*z)
            else:
                rospy.logwarn("Cannot find head message")

        return msg


    def to_voice_activity_msg(self, timestamp=None):
        """
        """
        is_speaking = self.is_speaking()
        msg = VoiceActivity()
        msg.person_id   = self.person_id
        msg.va_id       = self.person_id
        msg.is_speaking = is_speaking
        msg.is_speaking_confidence = 1.0
        # msg.is_last = False
        msg.turn_duration = rospy.Duration((self.get_turn_duration())/1e9)
        # if len(self.vfoa_buffer) > 0:

        if self.time_start_turn >= 0:
            vfoa = self.vfoa_buffer.reduce(t1=self.time_start_turn, t2=timestamp)

            if vfoa: # Function may be called while callback is populating the buffer
                for target, proba in vfoa.items():
                    msg.addresse.append(
                        IdWithProbability(str(target), float(proba)))

        return msg


    def to_face_info_msg(self, timestamp=None):
        """Return a FaceInfo message at time 'timestamp'. If 'timestamp' is
           None, return the lastest available.

        """
        msg = FaceInfo()
        msg.person_id = self.person_id
        msg.track_id  = self.track_id

        # fmsg = self.get_message_at(self.head_pose_buffer, timestamp)
        # TODO
        # tmsg = self.get_message_at(timestamp)
        #
        # if tmsg is not None:
        #     tmsg.age = 0
        #     tmsg.age_confidence = 0
        #     tmsg.gender = 0
        #     tmsg.gender_confidence = 0
        #     tmsg.smile_degree = 0
        #     tmsg.smile_degree_confidence = 0

        return msg


    def to_gaze_info_msg(self, timestamp=None):
        """Return a GazeInfo message at time 'timestamp'. If 'timestamp' is
           None, return the lastest available.

        """
        msg = GazeInfo()
        msg.person_id = self.person_id
        msg.track_id  = self.track_id

        # tmsg = self.get_message_at(timestamp)
        tmsg = self.get_message_at(self.tracklet_buffer, timestamp)

        if tmsg is not None:
            tmsg = tmsg[1]
            msg.head_gaze_available = 1
            msg.head_gaze = tmsg.head

            # Copy eye gaze for WP3 but is NOT eye gaze
            msg.eye_gaze_available = 1
            msg.eye_gaze = tmsg.head

            for t in tmsg.targets:
                if t.name == "Robot":
                    msg.probability_looking_at_robot = t.probability
                elif t.name == "Tablet":
                    msg.probability_looking_at_screen = t.probability

                ##################################################
                # Humavips
                # msg.attentions.append(IdWithProbability(str(t.name).replace("Person ", ""), t.probability))
                ##################################################

            vfoa = self.vfoa_buffer.reduce(t1=timestamp-1e9, t2=timestamp) # last 1 sec

            if vfoa: # Function may be called while callback is populating the buffer
                for target, proba in vfoa.items():
                    msg.attentions.append(
                        IdWithProbability(str(target), float(proba)))

        return msg
