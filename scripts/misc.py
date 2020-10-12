# coding=utf-8

import logging
import cv2

logger = logging.getLogger(__name__)


class BoundingBox(object):
    """
    """
    def __init__(self, pid, x, y, w, h, W=-1, H=-1):
        """Constructor

        Args:
            pid :
            x :
            y :
            w :
            h :
            W : image width in which the bounding box is defined
            H : image height in which the bounding box is defined

        """
        self.pid = pid
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.W = W
        self.H = H

    def __str__(self):
        return "{} ({},{}) {}x{} on {}x{}".format(
            self.pid, self.x, self.y, self.w, self.h, self.W, self.H)

    def __repr__(self):
        return self.__str__()

    def draw(self, image, color=(0,0,255), thickness=1):
        """Draw the bounding box on the image"""
        ratio = float(image.shape[0])/float(self.H) if self.H > 0 else 1.0
        x0 = int(ratio*self.x)
        y0 = int(ratio*self.y)
        x1 = int(ratio*(self.x + self.w - 1))
        y1 = int(ratio*(self.y + self.h - 1))
        cv2.rectangle(image, (x0, y0), (x1, y1), color=color, thickness=thickness)
        cv2.putText(image, str(self.pid), (x0, y0),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=color, thickness=1)



def load_mot_annotations(file_name):
    """Load a MOT challenge .csv file

    From https://motchallenge.net/instructions/

       <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

    is returned as

    {
      frame: [ BoundingBox(...), BoundingBox(...), ...]
      ...
    }

    for instance:

    {
      1: [ BoundingBox(3, ...), BoundingBox(8, ...), BoundingBox(7, ...) ],
      4: [ BoundingBox(3, ...), BoundingBox(7, ...) ],
      ...
    }

    """
    annotations = {}
    with open(file_name) as fid:
        for line in fid:
            line = line.strip()
            tok = line.split(",")
            if len(tok) != 10:
                logger.error("Line '{}' is not valid")
                continue
            frame = int(tok[0])
            pid = int(tok[1])
            bb_left = float(tok[2])
            bb_top = float(tok[3])
            bb_width = float(tok[4])
            bb_height = float(tok[5])
            conf = float(tok[6])
            x = float(tok[7])
            y = float(tok[8])
            z = float(tok[9])

            if frame not in annotations:
                annotations[frame] = []

            annotations[frame].append(
                BoundingBox(pid, bb_left, bb_top, bb_width, bb_height)
            )

    return annotations

def reassign_mot_id(input_mot, output_mot, t2p):
    """Reassign the IDs of the input file to the output file with
    providing map between old and new IDs

    Args:
        input_mot :
        output_mot :
        t2p : tracklet id to person id

    """

    with open(input_mot) as inp, open(output_mot, "w") as out:
        for line in inp:
            line = line.strip()
            tok = line.split(",")
            if len(tok) != 10:
                logger.error("Line '{}' is not valid")
                continue

            tid = int(tok[1])
            assert tid in t2p
            pid = t2p[tid]

            out.write("{},{},{},{},{},{},{},{},{},{}\n"
                      .format(tok[0], pid, tok[2], tok[3], tok[4], tok[5],
                              tok[6], tok[7], tok[8], tok[9]))


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
