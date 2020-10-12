#!/usr/bin/env python
import os
import sys
import cv2
import rospy
import datetime
# from collections import deque
from cv_bridge import CvBridge
from std_msgs.msg import ColorRGBA

from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3

from perception_msgs.msg import BodyJoints3D
from perception_msgs.msg import PersonTrackletArray
from perception_msgs.msg import GazeInfo
from perception_msgs.msg import GazeInfoArray
from perception_msgs.msg import TrackedPerson
from perception_msgs.msg import TrackedPersonArray
from perception_msgs.msg import FaceInfo
from perception_msgs.msg import FaceInfoArray

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from utils import str_time
from utils import get_3d_from_2d
from utils import get_3d_from_2d_area
from utils import get_joint_marker
import message_filters

import torch
import dlib
from pyopenface import prepare_open_face


openface = prepare_open_face()
