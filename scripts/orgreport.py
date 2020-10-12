######################################################################
# Copyright (c) 2018 Idiap Research Institute <http://www.idiap.ch/>
######################################################################

from __future__ import print_function

import os
import cv2

class OrgReport(object):
    """A class to export an org-mode report"""
    def __init__(self,
                 dir_name,
                 title = "TITLE",
                 options = "toc:nil"):
        """Constructor

        Args:
            dir_name :

        """
        self.dir_name = dir_name

        report_file = os.path.join(self.dir_name, "report.org")
        self.fid = open(report_file, "w")
        self.add_text("#+TITLE: {}\n".format(title))
        self.add_text("#+OPTIONS: {}\n".format(options))
        self.add_text("\n")


    def add_text(self, text):
        """Add some text"""
        self.fid.write(text)


    def add_image(self, image, file_name, sub_dir_name=".", caption=""):
        """Save an image and add it to the file

        The image will be saved in dir_name/sub_dir_name/file_name.

        """
        # dir_name/sub_dir_name
        absolute_dir_name = os.path.join(self.dir_name, sub_dir_name)
        if not os.path.exists(absolute_dir_name):
            os.mkdir(absolute_dir_name)

        #
        relative_image_path = os.path.join(sub_dir_name, file_name)
        cv2.imwrite(os.path.join(self.dir_name, relative_image_path), image)
        self.add_text("#+CAPTION: {}\n".format(caption))
        self.add_text("[[./{}]]\n".format(relative_image_path))
