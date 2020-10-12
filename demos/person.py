# coding: utf8

######################################################################
# Copyright (c) 2018 Idiap Research Institute <http://www.idiap.ch/>
######################################################################

class Person(object):
    """Hold the state of a person with useful attributes"""
    def __init__(self,
                 pid,
                 head_pose=None,
                 is_speaking=False):
        """
        """
        self.person_id = pid # An int
        self.is_speaking = is_speaking # A Boolean
        self.head_pose = head_pose # A ROS PoseStamped


    def __str__(self):
        x = 0.0
        y = 0.0
        z = 0.0
        if self.head_pose is not None:
            x = self.head_pose.pose.position.x
            y = self.head_pose.pose.position.y
            z = self.head_pose.pose.position.z
        s = "{} head {:.3f} {:.3f} {:.3f} speaks {}".format(self.person_id,
                                                            x, y, z,
                                                            self.is_speaking)
        return s
