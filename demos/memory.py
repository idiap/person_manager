# coding: utf8

######################################################################
# Copyright (c) 2018 Idiap Research Institute <http://www.idiap.ch/>
######################################################################

class Memory(object):
    """Hold the memory of past and present events"""
    def __init__(self):
        """
        """
        # Interacting persons { pid : Person(), ... }
        self.front_persons = {}

        # People met from the beginning { pid : Person(), ... }
        self.known_persons = {}

    def __str__(self):
        """
        """
        s = "Front {}".format(len(self.front_persons))
        return s
