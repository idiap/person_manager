# coding=utf-8

import os
import sys
import logging

import cv2

logger = logging.getLogger(__name__)

class ImageProvider(object):
    """
    """
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def get_indices(self):
        """To facilitate access when indices are not continuous"""
        return list(range(len(self)))


class ImageListProvider(ImageProvider):
    """Input is a .txt file containing absolute path to images"""
    def __init__(self, image_list):
        """
        """
        self.file_names= []

        with open(image_list) as fid:
            for image_name in fid:
                image_name = image_name.strip()
                if os.path.isfile(image_name):
                    self.file_names.append(image_name)
                else:
                    logger.error("{} is not a file".format(image_name))

        logger.error("Loaded {} images".format(len(self)))

    def __getitem__(self, index):
        return cv2.imread(self.file_names[index])

    def __len__(self):
        return len(self.file_names)


class NumberNameImageListProvider(object):
    """Input is a .txt file listing the images, and the name of the images
    is a number which will be used to access/identify them (for
    instance their timestamp in nanoseconds).

    It could be 1.jpg, 10.jpg, 11.jpg, 200.jpg, etc. and we will do
    provider[1], provider[10], provider[11], provider[200], etc.

    """

    def __init__(self, image_list):
        """
        """
        self.file_names= {} # { timestamp: file_name, 123123:/path/to/123123.jpg }

        with open(image_list) as fid:
            for file_name in fid:
                file_name = file_name.strip()
                timestamp = os.path.split(file_name)[1]
                timestamp = os.path.splitext(timestamp)[0]
                timestamp = int(timestamp)
                if os.path.isfile(file_name):
                    self.file_names[timestamp] = file_name
                else:
                    logger.error("{} is not a file".format(file_name))

        logger.error("Loaded {} images".format(len(self)))

    def __getitem__(self, number):
        return cv2.imread(self.file_names[number])

    def __len__(self):
        return len(self.file_names)

    def get_indices(self):
        """
        """
        indices = list(self.file_names.keys())
        indices.sort()
        return indices
