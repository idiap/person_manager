# coding=utf-8

######################################################################
# Copyright (c) 2018 Idiap Research Institute, <http://www.idiap.ch>
######################################################################

import cv2
import copy
import rospy
import logging
import datetime
import numpy as np

from visualization_msgs.msg import Marker

from geometry_msgs.msg import Pose

from perception_msgs.msg import BodyJoints3D

from collections import deque

logger = logging.getLogger(__name__)

TIME_FORMAT = "%Y-%m-%d@%H:%M:%S.%f"

COLORS = [ (128,128,128),
           ( 30, 45,190), ( 30,110,240), ( 70,200,235),
           (110,185,160), (185,150, 20), ( 85, 60, 10) ]


class StampedData(object):
    """Hold data with an associated timestamped

    The data can be:

      - values: 3.14
      - lists: [3.14, 0.07, ...]
      - numpy arrays: np.array([3.14, 0.07, ...])
      - dict: {"key1": value/list/array, "key2": value/list/array}

    but should be consistent with the first elemtn inserted (not
    checked). List/array/value are converted to np.array to ease
    computation of mean.

    Args:
        maxlen: Length of deque holding the data

    Example:
        >>> d.add(t1, 3.14)
        >>> d.add(t2, 0.07)

    """
    def __init__(self, maxlen=1000):
        """
        """
        self.maxlen = maxlen
        self.data = deque(maxlen=self.maxlen);
        self.is_dict = None

    def __len__(self):
        return len(self.data)

    def __str__(self):
        s = ""
        for i, (t, d) in enumerate(self.data):
            if i > 0:
                s += "\n"
            s += "{}: {}".format(t, d)
        return s

    def add(self, timestamp, data):
        """
        """
        if self.is_dict is None:
            self.is_dict = True if isinstance(data, dict) else False

        if self.is_dict:
            self.data.append((timestamp, data))
        else:
            self.data.append((timestamp, np.array(data)))
        # if self.typ is None:
        #     self.typ = type(data)
        # if isinstance(date, self.typ):
        #     self.data.append((timestamp, data))
        # else:
        #     logger.error("Expect element of same type")

    def get(self, t1=0, t2=1e300):
        """Return the data between t1 and t2

        Args:
            reduction: (None | mean | sum)
        """
        data = []
        c = copy.deepcopy(self.data) # Avoid "RuntimeError: deque mutated during iteration"
        for t, d in c:
            if t1 <= t and t <= t2:
                data.append((t, d))

        return data

    def reduce(self, t1=0, t2=1e300, reduction="mean"):
        """Return the mean in the same type (list, numpy array, float, etc.)
        as the data

        """
        data = self.get(t1, t2)

        if len(data) == 0:
            return None

        elif self.is_dict:
            return self._reduce_dict(data, reduction)

        else:
            return self._reduce(data, reduction)

    def _reduce(self, data, reduction="mean"):
        """Input is (timestamp, not_dict). The not_dict should have already
        been converted to numpy array

        """
        s = None

        for _, d in data:
            if s is None:
                s = d
            else:
                s += d

        if reduction == "mean":
            s = s/float(len(data))

        return s

    def merge(self, other):
        """
        """
        for t, v in other.data:
            self.add(t, v)

        # Sort, but should be sorted already if other is later that
        # self
        l = list(self.data)
        s = sorted(l, key=lambda x: x[0])
        self.data = deque(maxlen=self.maxlen)
        for t, v in s:
            self.add(t, v)

    def update_key(self, old_key, new_key):
        """If the data is of type 'dict', update old_key name by new_key name.

        Useful when re-identification is performed. old_key and
        new_key should not be in the dict at the same time (not
        checked).

        """
        if self.is_dict:
            for _, d in self.data:
                if old_key in d:
                    d[new_key] = d.pop(old_key)

    def _reduce_dict(self, data, reduction="mean"):
        """Input is (timestamp, dict)"""
        accum = {}
        for _, d in data:
            for k, v in d.items():
                if k not in accum:
                    accum[k] = []
                accum[k].append(np.array(v))

        for k in accum:
            if reduction == "mean":
                N = len(accum[k])
                accum[k] = sum(accum[k]) / float(N)
            elif reduction == "sum":
                accum[k] = sum(accum[k])

        return accum


def get_closest_rosmsg(time_ns, iterable):
    """
    Return message which is closest in time to time_ns.
    The message may be in the past or in the future.
    It can also have the same timestamp.
    """
    if len(iterable) == 0:
        return None, 0

    mini = 0
    closest_msg = None
    closest_time = 0

    for i, msg in enumerate(list(iterable)):
        # Converted to list to avoid "deque mutated during iteration"
        msg_ns = msg.header.stamp.to_nsec()
        sdiff = msg_ns - time_ns
        adiff = abs(sdiff)

        if (i == 0) or (adiff < mini):
            mini = adiff
            closest_msg = copy.deepcopy(msg)
            closest_time = sdiff

    return closest_msg, closest_time


def str_time(sec=0, usec=0, nsec=0):
    """
    """
    return datetime.datetime.fromtimestamp \
        (float(sec) + float(1e-6)*usec + float(1e-9)*nsec). \
        strftime(TIME_FORMAT)





def get_joint_marker(marker_id, x, y, z):
    """
    """
    mrk = Marker()
    mrk.id = marker_id
    mrk.type = mrk.SPHERE
    mrk.action = mrk.ADD
    mrk.scale.x = 0.05
    mrk.scale.y = 0.05
    mrk.scale.z = 0.05
    mrk.color.r = 1.0
    mrk.color.g = 0.0
    mrk.color.b = 0.0
    mrk.color.a = 1.0
    mrk.pose.position.x = x/1000.0
    mrk.pose.position.y = y/1000.0
    mrk.pose.position.z = z/1000.0
    mrk.pose.orientation.x = 0.0
    mrk.pose.orientation.y = 0.0
    mrk.pose.orientation.z = 0.0
    mrk.pose.orientation.w = 1.0
    mrk.lifetime = rospy.Duration(0.25)
    return mrk


def draw_bounding_box(image, x0, y0, x1, y1,
                      color=(128,128,128), width=3,
                      ID=-1,
                      name="?",
                      shape="rect",
                      thickness=3):
    """Draw a rectangle and some text above it.

    The text above the rectangle has following format "<ID> | <name>"

    Args:
        image :
        x0    :
        y0    :
        x1    :
        y1    :
        color :
        width :
        ID    :
        shape : "rect", "circle"

    """
    if shape == "circle":
        xm = (x1-x0)/2
        ym = (y1-y0)/2
        cv2.circle(image,
                   (int(x0 + xm), int(y0 + ym)),
                   int(xm),
                   color,
                   thickness)
        cv2.circle(image,
                   (int(x0 + xm), int(y0 + ym)),
                   int(xm),
                   (0,255,255),
                   thickness/2)

    else: # shape == "rect":
        cv2.rectangle(image,
                      (int(x0), int(y0)),
                      (int(x1), int(y1)), color,
                      thickness)
    if ID>0:
        text = "{} | {}".format(ID, name)
        cv2.putText(image, text, (x0,y0 - 2*thickness),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=color,
                    thickness=3)
        cv2.putText(image, text, (x0,y0 - 2*thickness),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255-color[0],255-color[1],255-color[2]),
                    thickness=1)


def draw_facial_landmarks(image,
                          landmarks,
                          x_offset, y_offset,
                          color=(0,255,0)):
    """Draw dlib facial landmarks on input image.

    Args:
        image     :
        landmarks : List of tuples (x, y)
        x_offset  : x offset to add (if landmarks were computed on a crop)
        y_offset  :
        color     :

"""
    for (x,y) in landmarks:
        cv2.circle(image, (y_offset + x, x_offset + y), 1, color)


def body_joints_to_list(joints):
    """Concatenate all joints to a list to iterate easily.

    Args:
        joints : A perception_msgs/BodyJoints2D or
                 a perception_msgs/BodyJoints3D message

    Returns:
        List of the elements inside joints (in the same order as in
        BodyJoints2D or BodyJoints3D).

    """
    return [ joints.nose, joints.neck,
             joints.leye, joints.reye,
             joints.lear, joints.rear,
             joints.lshoulder, joints.rshoulder,
             joints.lelbow, joints.relbow,
             joints.lwrist, joints.rwrist,
             joints.lhip, joints.rhip,
             joints.lknee, joints.rknee,
             joints.lankle, joints.rankle,
    ]



def draw_body_joints_2d(image, msg, color):
    """Draw the BodyJoints2D on input image"""
    W = float(image.shape[1])/float(msg.image_width)
    H = float(image.shape[0])/float(msg.image_height)

    joints = body_joints_to_list(msg)

    for j in joints:
        if j.c > 0:
            x = j.w*W
            y = j.h*H
            cv2.circle(image, (int(x),int(y)), 5, color, -1)


# TODO: Function to remove from here and put to misc
def scale_bounding_box(image,
                       h, w, height, width,
                       image_height=-1, image_width=-1):
    """Adapt the coordinates of the input bounding box.


    Args:
        image  : Image in which to project the bounding box.
        h      :
        w      :
        height :
        width  :
        image_height : Height of image in which the bounding box is.
        image_width  :

    Returns:
        x0, x1, y0, y1 : Start and end points of bounding box

    """
    if image_height < 0:
        image_height = image.shape[0]
    if image_width < 0:
        image_width = image.shape[1]
    W = float(image.shape[1])/float(image_width)
    H = float(image.shape[0])/float(image_height)
    x0 = int(w*W)
    y0 = int(h*H)
    x1 = int((w + width)*W)
    y1 = int((h + height)*H)
    return x0, x1, y0, y1


def project_2d_to_3d(depth, u, v, fx, fy, cx, cy):
    """Return the 3D coordinate corresponding to point (u,v) in depth
    image.

    Args:
        depth :
        u     :
        v     :
        fx    :
        fy    :
        cx    :
        cy    :

    Returns:
        x3d, y3d, z3d : 3D coordinates of input (u, v)

    """
    x3d = 0
    y3d = 0
    z3d = 0
    if depth is not None:
        x3d = (u - cx) * depth[int(v)][int(u)] / fx
        y3d = (v - cy) * depth[int(v)][int(u)] / fy
        z3d = depth[int(v)][int(u)]
    return x3d, y3d, z3d


def project_2d_to_3d_area(depth, u, v, fx, fy, cx, cy, n=2):
    """Return the 3D coordinate corresponding to point (u,v) in depth
    image average in the vicinity of (u, v) with a radius of 'n'
    (square).

    Args:
        depth :
        u     :
        v     :
        fx    :
        fy    :
        cx    :
        cy    :
        n     : Side of square is (2*n + 1) around (u, v)

    Returns:
        x3d, y3d, z3d : Averaged 3D coordinates of around (u, v)

    """
    H = depth.shape[1]
    W = depth.shape[0]

    umin = max(int(u)-n, 0)
    umax = min(int(u)+n+1, W-1)
    vmin = max(int(v)-n, 0)
    vmax = min(int(v)+n+1, H-1)

    x3d = 0
    y3d = 0
    z3d = 0

    if depth is not None:
        roi = depth[int(vmin):int(vmax),int(umin):int(umax)]

        z3d = np.mean(roi[np.nonzero(roi)])

        if np.isnan(z3d):
            z3d = 0

        if z3d > 0:
            x3d = (u - cx) * z3d / fx
            y3d = (v - cy) * z3d / fy

    # print("({},{}) [{} {} {} {}] - {:.2f} {:.2f} {:.2f}". \
    #       format(u,v,umin,umax,vmin,vmax,x3d,y3d,z3d))

    return x3d, y3d, z3d


def project_head_on_depth(depth, msg,
                          depth_fx, depth_fy,
                          depth_cx, depth_cy,
                          area):
    """Project the head on the depth image.

    Project only the msg.head.position on the depth image. The
    msg.head.orientation is preserved.

    If the depth is None, it returns the msg.head (geometry_msgs/Pose)
    without modification whose location is already estimated from head
    height.

    Args:
        depth    :
        msg      : A perception_msgs/BodyJoints2D message
        depth_fx :
        depth_fy :
        depth_cx :
        depth_cy :
        area     : Side of neighbourhood around joint

    Returns:
        A geometry_msgs/Pose message

    """
    pose = Pose()
    pose.position.x = msg.head.position.x
    pose.position.y = msg.head.position.y
    pose.position.z = msg.head.position.z
    pose.orientation.x = msg.head.orientation.x
    pose.orientation.y = msg.head.orientation.y
    pose.orientation.z = msg.head.orientation.z
    pose.orientation.w = msg.head.orientation.w


    if depth is not None:
        # Keep quaternion, but use depth for head location
        W = float(depth.shape[1])/float(msg.box.image_width)
        H = float(depth.shape[0])/float(msg.box.image_height)

        # Compute centre of bounding box
        u_head = msg.box.w + msg.box.width/2
        v_head = msg.box.h + msg.box.height/2
        u_head = u_head*W
        v_head = v_head*H

        # Project centre on depth
        x3d, y3d, z3d = project_2d_to_3d_area(depth,
                                              u_head, v_head,
                                              depth_fx,
                                              depth_fy,
                                              depth_cx,
                                              depth_cy,
                                              area)
        # Convert to metres if data available
        pose.position.x = x3d/1000.0
        pose.position.y = y3d/1000.0
        pose.position.z = z3d/1000.0

    return pose


def project_2d_body_joints_on_depth(image, depth, msg,
                                    depth_fx, depth_fy,
                                    depth_cx, depth_cy,
                                    area):
    """Project the 2D joints on input depth.

    Args:
        image    :
        depth    :
        msg      : A perception_msgs/BodyJoints2D message
        depth_fx :
        depth_fy :
        depth_cx :
        depth_cy :
        area     : Side of neighbourhood around joint

    Returns:
        A perception_msgs/BodyJoints3D (coordinates in meters).

    """
    W = float(image.shape[1])/float(msg.joints.image_width)
    H = float(image.shape[0])/float(msg.joints.image_height)

    omsg = BodyJoints3D()
    j2d = body_joints_to_list(msg.joints)
    j3d = body_joints_to_list(omsg)

    for i in range(len(j2d)):
        if j2d[i].c > 0:
            x = j2d[i].w*W
            y = j2d[i].h*H
            x3d, y3d, z3d, = project_2d_to_3d_area(depth, x, y,
                                                   depth_fx, depth_fy,
                                                   depth_cx, depth_cy, area)

            j3d[i].x = x3d/1000.0
            j3d[i].y = y3d/1000.0
            j3d[i].z = z3d/1000.0

    return omsg


def montage(image_list, height=100):
    """Concatenate input images on a single row given height"""
    trombi = None

    for image in image_list:
        H = image.shape[0]
        W = image.shape[1]
        f = float(W)/float(H)
        new_size = (int(height*f), height)
        add = cv2.resize(image, new_size)

        if trombi is None:
            trombi = add
        else:
            trombi = np.hstack((trombi, add))

    return trombi


# def get_color(person_id):
#     """
#     """
#     return COLORS[1 + person_id % (len(COLORS)-1)]


def get_color(i, N=10, M=255, opencv=True):
    """
    Args:
        i: Color number
        N: Number of colors to split range in (80 for COCO)
        M: Maximum value of color (255 or 1.0 usually)
        cv2: Return BGR for OpenCV
    """
    def c(x, a, b=1.0, s=0.0):
        return max(s, min(b, s + (b-s)*(2 - abs(x)/a)))
    a = N/6.0
    r = M - c(i-3*a, a, M)
    g = c(i - 2*a, a, M)
    b = c(i - 4*a, a, M)
    rgb = [b, g, r] if opencv else [r, g, b]
    return tuple(map(type(M), rgb))

def draw_probabilities(image, probabilities, x0, y0, x1, y1, h=5):
    """Draw

    Args:
        probabilities: Dict with {target: proba}
        h: height of the bar
    """
    offset = 5

    s = sum([p for _, p in probabilities.items()])

    W = x1 - x0

    lengths = { name: int(W*p/s) for name, p in probabilities.items() }

    xa = x0
    xb = x0
    for target, length in lengths.items():
        color = (255, 255, 255)
        if target.lower() in ["unfocused", "aversion"]:
            color = (255, 255, 255)
        elif target.lower() in ["robot", "camera"]:
            color = (0,0,0)
            # color = (103,228,103) # Humavips color
        else:
            try:
                i = int(target)
                color = get_color(i)
            except Exception as e:
                color = (113,113,255)


        xb = xa + length
        cv2.rectangle(image, (xa, y1+offset), (xb, y1+offset+h), color=color, thickness=-1)
        xa = xb
