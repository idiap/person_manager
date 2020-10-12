# coding=utf-8

import os
import sys
import logging
import argparse

import cv2

from misc import load_mot_annotations
from misc import reassign_mot_id

# from tracklet_manager import TrackletManager
from provider import NumberNameImageListProvider
from tracklet_manager import TrackletManager

logger = logging.getLogger(__name__)

WINDOW_NAME = "Image"

if __name__ == "__main__":
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image", type=str, default="", required=True,
        help="Path to input images/video")
    parser.add_argument(
        "-a", "--annotations", type=str, default="", required=True,
        help="Path to input tracklets")
    parser.add_argument(
        "--fps", type=int, default=15,
        help="Speed passed to waitKey when pressing space")
    parser.add_argument(
        "--skip", type=int, default=10,
        help="How many frames to skip when pressing [ or ]")
    parser.add_argument(
        "--dlib-landmarks", type=str, default="",
        help="Path to shape_predictor_68_face_landmarks.dat")
    parser.add_argument(
        "--record", type=str, default="",
        help="Path of directory to record")
    parser.add_argument(
        "--visu", help="Whether to display", action="store_true")
    parser.add_argument(
        "--pwd", type=str, default="/tmp",
        help="Path of working directory")
    parser.add_argument(
        "--verbose", type=int, default=20,
        help="Logging verbosity (10, 20, 30)")
    # parser.add_argument("args", nargs=argparse.REMAINDER)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.pwd):
        os.makedirs(args.pwd)

    logging.basicConfig(format="[%(name)s] %(message)s", level=args.verbose)

    tm = TrackletManager(dlib_landmarks=args.dlib_landmarks)

    annotations = load_mot_annotations(args.annotations)

    image_provider = NumberNameImageListProvider(args.image)
    indices = image_provider.get_indices()

    if args.visu:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    n_images = len(image_provider)
    i_image = 0
    pause = 0
    prev_pause = args.fps # When we changed its value with +/-
    while True:
        index = indices[i_image]
        logger.info("Loading image {} {}/{}"
                    .format(index, i_image, n_images))

        image = image_provider[index]
        bbs = annotations[index] if index in annotations else []

        new_bbs = tm.manage(image, bbs, index)

        display = image.copy()
        for bb in new_bbs:
            bb.draw(display)

        if args.record:
            cv2.imwrite(os.path.join(args.record, name), display)

        key = ""
        if args.visu:
            # print("if args visu")
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(pause) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if pause > 0: prev_pause = pause
            pause = 0 if pause > 0 else prev_pause
        elif key == ord('a'):
            i_image = 0
            pause = 0
        elif key == ord('p'):
            i_image -= 1
            pause = 0
        elif key == ord('n'):
            i_image += 1
            pause = 0
        elif key == ord('['):
            i_image -= args.skip
            pause = 0
        elif key == ord(']'):
            i_image += args.skip
            pause = 0
        elif key == ord('-'):
            prev_pause = pause
            pause -= 5
        elif key == ord('+'):
            prev_pause = pause
            pause += 5
        elif key == ord('e'):
            i_image = n_images - 1
            pause = 0

        if pause < 0:
            pause = 0

        if pause > 0 or args.record or not args.visu:
            i_image += 1

        if i_image < 0:
            i_image = 0

        if i_image >= n_images:
            logger.info("Reaching the end, now pausing")
            i_image = n_images - 1
            pause = 0
            if not args.visu: break

    if args.visu:
        cv2.destroyAllWindows()

    output_name = os.path.split(args.annotations)[0]
    output_name = os.path.join(output_name, "reid.txt")
    logger.info("Saving to {}".format(output_name))
    reassign_mot_id(args.annotations, output_name, tm.tracklet_id_to_person_id)

    # id = TrackletManager(
    #     dlib_landmarks_file_name=args.dlib_landmarks,
    # )

    # for i in range(len(image_provider)):
    #     image = image_provider[i]
    #     display = image.copy()

    #     if i in annotations:
    #         for bb in annotations[i]:
    #             bb.draw(display)

    #         tm.manage(image, annotations[i])

    #     cv2.imshow(WINDOW_NAME, display)
    #     key = cv2.waitKey(30) & 0xFF

    #     if key == ord('q'):
    #         break

    # cv2.destroyAllWindows()
