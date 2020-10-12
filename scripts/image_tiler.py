from __future__ import print_function

######################################################################
# Copyright (c) 2018 Idiap Research Institute, <http://www.idiap.ch>
######################################################################

import cv2
import numpy as np


class ImageTiler(object):
    """An ImageTiler makes a montage of severat image into one image.

    The number of rows or cols has to be specified as well at the
    width or height the images should be resized to.

    """
    def __init__(self,
                 tiles_per_row=10,
                 size=(-1, 100)):
        """Constructor

        Args:
            tiles_per_row :
            size          : Size at which images should be resized.
        """
        self.tiles_per_row = tiles_per_row

        self.w_tile = size[0]
        self.h_tile = size[1]

        # Iterator positions for the next image to be added.
        self.i_rows = -1
        self.i_cols = -1

        self.montage = None
        self.empty_row = None


    def _increment(self):
        """
        """
        # Since we call _increment before adding an image, we do
        # nothing the first time
        if self.i_rows < 0 and self.i_cols < 0:
            self.i_rows = 0
            self.i_cols = 0
            return

        self.i_cols += 1
        if self.i_cols == self.tiles_per_row:
            self.i_cols = 0
            self.i_rows += 1
            # Add new black line below
            self.montage = np.vstack((self.montage, self.empty_row))


    def _initialise(self, image):
        """
        """
        H = image.shape[0]
        W = image.shape[1]
        if self.w_tile > 0:
            self.w_tile = int(self.w_tile)
            self.h_tile = int(float(self.w_tile)/float(W)*float(H))
        else:
            self.h_tile = int(self.h_tile)
            self.w_tile = int(float(self.h_tile)/float(H)*float(W))

        dim = (self.h_tile, self.tiles_per_row*self.w_tile, 3)
        self.empty_row = np.zeros(dim, np.uint8)
        self.montage = np.copy(self.empty_row)

    def add_image(self, image):
        """
        """
        # The first time, compute everything, assuming the following
        # image will be roughly of the same size (to determine w_tile
        # and h_tile)
        if self.montage is None:
            self._initialise(image)

        # Increment to get coordinate of new image, and potentially
        # append new line below
        self._increment()

        x0 = self.i_cols*self.w_tile
        y0 = self.i_rows*self.h_tile
        x1 = x0 + self.w_tile
        y1 = y0 + self.h_tile

        self.montage[y0:y1, x0:x1] = cv2.resize(image,
                                                (self.w_tile, self.h_tile))


    def add_images(self, images):
        """Call add_image on all input image

        Args:
            images : List of images

        """
        for image in images:
            self.add_image(image)
