# coding=utf-8

import os
import sys
import copy
import logging
import argparse

import cv2

from pytopenface import OpenfaceComputer
from pytopenface import enlarge_bounding_box
from pytopenface import Reidentifier

from person import Person

from misc import scale_bounding_box

logger = logging.getLogger(__name__)

class TrackletManager(object):
    """Class to manage tracklets and re-identification"""
    def __init__(self,
                 dlib_landmarks="",
                 use_cuda=1,
                 min_votes_for_reid=10,
                 keep_threshold=0.25,
                 reid_threshold=0.4,
                 max_nb_features_per_id=200):
        """
        """
        # Persons list
        #
        #  - known_persons: All known persons from the beginning of
        #    time
        #
        #  - front_persons: Persons who are currently interacting with
        #    the robot, they are not necessarily visible, if they are
        #    speaking outside the field of view. This is a subset of
        #    'known_persons': when a person does not interact with the
        #    robot anymore, she is removed from the 'front_persons'
        #    list, but remains in the 'known_persons'.
        #
        self.known_persons = []
        self.front_persons = []
        # self.persons = []

        self.max_length_buffer = 200

        # The current persons which we focus our attention on
        self.person_of_interest = []

        ############################################################
        # Reidentification part
        ############################################################
        # Minimum number of votes for an identity to:
        #  - re-identify an non-identified tracklet
        #  - merge to tracklets
        self.min_votes_for_reid = min_votes_for_reid

        # Used when reidentification is performed
        self.tracklet_id_to_person_id = {}

        # Lists of tracklets that should not be merged together
        # because obviously being of different persons.
        #
        # For instance, if mutually_exclusive is [[1,2,4], [3,2]]
        # (which would mean that persons 1,2,4 have been seen together
        # on the smae image, and later persons 2,3 also) and that for
        # some reason (e.g. occlusion), the merge process wants to
        # merge [1,2], then this is forbidden.
        self.mutually_exclusive = set([])

        # A dictionnary to accumulate votes for each tracklet. When
        # the number of votes is larger than self.min_votes_for_reid
        self.tracklet_votes = {}

        self.openface_computer = OpenfaceComputer(dlib_landmarks,
                                                  useCuda=use_cuda)

        self.reidentifier = Reidentifier(keep_threshold=keep_threshold,
                                         reid_threshold=reid_threshold,
                                         max_nb_features_per_id=max_nb_features_per_id)


    def manage(self, color, bbs, ts):
        """
        Args:
            color : Input color image
            bbs   : List of bounding boxes to re-identify
            ts    : Corresponding timestamp of the image (in nanoseconds)
        """
        # logger.info("Managing tracklet at {}".format(ts))

        # List IDs that are present at the same time
        visible_ids = set([])
        present_ids = [bb.pid for bb in bbs]

        # Create a new person if not already present
        # for tracklet_msg in imsg.data:
        for bb in bbs:
            ID = bb.pid
            if ID not in self.tracklet_id_to_person_id:
                logger.info("Create person {}".format(ID))
                self.tracklet_id_to_person_id[ID] = ID

        ######################################################################
        # DEBUG
        ######################################################################
        # Simply fill front_persons with the tracklets, no
        # reidentification is performed
        ######################################################################
        # self.front_persons = []

        # for tracklet_msg in imsg.data:
        #     tid = tracklet_msg.tracklet_id
        #     pid = self.tracklet_id_to_person_id[tid]

        #     p = self.get_person_with_id(pid)

        #     if p is None:
        #         p = Person(self.max_length_buffer)
        #         self.known_persons.append(p)
        #     else:
        #         print("Got {}".format(pid))

        #     p.update(tracklet_msg)
        #     p.person_id = pid
        #     self.front_persons.append(p)

        # print("Known {} Front {}".format(len(self.known_persons),
        #                                  len(self.front_persons)))
        ######################################################################

        # Make pair of IDs that are mutually exclusive to avoid
        # merging 2 of them
        # print("present_id {}".format(present_ids))
        # print("self.mutually_exclusive {}".format(self.mutually_exclusive))
        n_ids = len(present_ids)
        for i in range(0, n_ids-1):
            for j in range(i+1, n_ids):
                # Tracklet ID and person ID for first element i
                tidi = present_ids[i]
                pidi = self.tracklet_id_to_person_id[tidi]
                # Tracklet ID and person ID for first element j
                tidj = present_ids[j]
                pidj = self.tracklet_id_to_person_id[tidj]

                # print("tidi {} pidi {} tidj {} pidj {}".format(tidi, pidi, tidj, pidj))
                self.add_mutually_exclusive(tidi, tidj)
                self.add_mutually_exclusive(pidi, tidj)
                self.add_mutually_exclusive(tidi, pidj)
                self.add_mutually_exclusive(pidi, pidj)
        # print("self.mutually_exclusive {}".format(self.mutually_exclusive))

        # if self.reidentifier.features is not None:
        #     print("gallery {}".format(self.reidentifier.features.shape))

        # print("Votes {}".format(self.tracklet_votes))

        # Vote for ids
        lists_to_merge = []
        for bb in bbs:
            x0, x1, y0, y1 = scale_bounding_box(color,
                                                bb.y, bb.x,
                                                bb.h, bb.w,
                                                bb.H, bb.W)
            x2, x3, y2, y3 = enlarge_bounding_box(x0, x1, y0, y1, color, 0.1)

            # Extract face (no copy)
            crop = color[y2:y3,x2:x3]

            features = self.openface_computer.compute_on_image(crop, 1)

            # if features.face is not None:
            #     cv2.imshow("visu", features.face)

            tid = bb.pid
            pid = self.tracklet_id_to_person_id[tid]

            # if features.face is not None and features.visu is not None:
            #     trombi = montage([crop, features.visu, features.face])
            #     # cv2.imshow("Trombi", trombi)
            #     # cv2.waitKey(1)
            #     if self.report is not None:
            #         sub_dir = "{}".format(rostime)
            #         image_name = "{}_pid_{}_tid_{}.jpg".format(rostime, pid, tid)
            #         self.report.add_image(trombi, image_name, sub_dir)

            # rospy.loginfo("Tracklet {} as {}".format(tid, pid))

            if features.features is not None:
                self.reidentifier.add(features.features, pid, crop)
                reid = self.reidentifier.identify(features.features)
                # print("pid {} tid {} votes {}".format(pid, tid, reid.votes))
                self.accumulate_votes(reid.votes, pid, tid)

            # Perform merging of identities
            for tracklet_id, votes_per_id in self.tracklet_votes.items():
                ids_to_merge = set([tracklet_id])
                for other_id, n_votes in votes_per_id.items():
                    if n_votes >= self.min_votes_for_reid:
                        ids_to_merge.add(other_id)
                if len(ids_to_merge) > 1:
                    lists_to_merge.append(list(ids_to_merge))

            # Get freshly updated person ID
            # pid = self.tracklet_id_to_person_id[tid]
            p = self.get_person_with_id(pid)
            # print("PERSON {}/{}".format(pid,tid))

            if p is None:
                p = Person(self.max_length_buffer)
                self.known_persons.append(p)

            if features.features is not None:
                p.is_identified += 1

            # joints3d = BodyJoints3D()

            p.person_id = pid
            p.track_id  = tid

        # print("status {}".format(self.reidentifier.get_status()))

        # Prepare IDs to merge.
        old_to_new = {}
        for list_to_merge in lists_to_merge:
            new_id = min(list_to_merge)
            for theid in list_to_merge:
                if theid not in old_to_new:
                    old_to_new[theid] = new_id
                else:
                    old_to_new[theid] = min(new_id, old_to_new[theid])

        # Recursively update the 'old_to_new'. If 5 should be
        # reidentified as 4, and 4 as 3, then 5 should be reidentified
        # as 3.
        n_changes = 1
        while n_changes > 0:
            n_changes = 0
            for old, new in old_to_new.items():
                if old_to_new[new] != new:
                    n_changes += 1
                    old_to_new[old] = old_to_new[new]

        ids_to_merge = {}
        for old, new in old_to_new.items():
            if new not in ids_to_merge:
                ids_to_merge[new] = set([])
            ids_to_merge[new].add(old)

        for new_id in ids_to_merge:
            self.merge_persons(new_id, list(ids_to_merge[new_id]))

        new_bbs = copy.deepcopy(bbs)
        for bb in new_bbs:
            bb.pid = self.tracklet_id_to_person_id[bb.pid]

        # When all persons are reidentified, we update
        # 'front_persons'. We don't do it at the same time of
        # reidentification because this callback runs in parallel of
        # the 'run' main loop and it can cause some access
        # conflicts. And reidentification is time consuming, so by the
        # time the 'run' accesses the 'front_persons', it might be
        # emptied by the callback.
        self.front_persons = []
        for bb in bbs:
            tid = bb.pid
            pid = self.tracklet_id_to_person_id[tid]

            # print("{} {}".format(tid, pid))
            p = self.get_person_with_id(pid)
            if p is None:
                logger.warning("Could not find person {}".format(pid))
            else:
                self.front_persons.append(p)

        # print(self.tracklet_id_to_person_id)


        return new_bbs



    def are_mutually_exclusive(self, id1, id2):
        """Return if (id1,id2) is in self.mutualy_exclusive"""
        if (min(id1,id2), max(id1,id2)) in self.mutually_exclusive:
            return True
        else:
            return False

    def add_mutually_exclusive(self, id1, id2):
        """Add a new tuple to 'mutually_exclusive'"""
        # if id1 != id2:
        self.mutually_exclusive.add((min(id1,id2), max(id1,id2)))

    def get_person_with_id(self, ID):
        """Return the person with input ID, and return None if does not exist.

        """
        for p in self.known_persons:
            if p.person_id == ID:
                return p
        return None

    def accumulate_votes(self, votes, person_id, tracklet_id):
        """Accumulate votes to 'tracklet_votes' and take into account the
           'mutually_exclusive' pairs

        """
        # print("before accum {}".format(self.tracklet_votes))
        # print("mutually_exclusive {}".format(self.mutually_exclusive))
        # Create if needed
        if person_id not in self.tracklet_votes:
            self.tracklet_votes[person_id] = {}

        accumulated = self.tracklet_votes[person_id]

        for k,v in votes.items():
            p1 = (min(tracklet_id, k), max(tracklet_id, k))
            p2 = (min(person_id,   k), max(person_id,   k))

            # Don't accumulate when mutually exclusive
            if (p1 not in self.mutually_exclusive) and \
               (p2 not in self.mutually_exclusive):
                if k not in accumulated:
                    accumulated[k] = 0
                accumulated[k] += v

        # print("after accum {}".format(self.tracklet_votes))

    def merge_persons(self, new_id, ids_to_merge):
        """Merge 'ids_to_merge' into 'new_id'

           This is a convenient function to call to avoid changing
           directly the __tracklet_cb() part where the merging is
           actually done (if later we add reidentifier based on
           clothes, etc.)

        """
        logger.info("Merging identities {} into {}"
                    .format(ids_to_merge, new_id))

        # Update correspondance between tracklet id and person id
        for theid in ids_to_merge:
            self.tracklet_id_to_person_id[theid] = new_id

        for other in ids_to_merge:
            if other in self.person_of_interest:
                self.person_of_interest = [new_id]
                logger.info("Person of interest is now {}".format(new_id))

        # Propagate ids to deleted tracklet.
        #
        # For instance, if person 28 exists, and if tracklet 30 was
        # reidentified as 29, then we have
        #
        #   tracklet_id_to_person_id = {28: 28, 29: 29, 30: 29}
        #
        # If now we identify 29 as being 28, then we also have to
        # propagate 30 to 28, and get
        #
        #   tracklet_id_to_person_id = {28: 28, 29: 28, 30: 28}
        #
        for tid, pid in self.tracklet_id_to_person_id.items():
            if pid in ids_to_merge:
                self.tracklet_id_to_person_id[tid] = new_id

        # print("tracklet_id_to_person_id {}".format(self.tracklet_id_to_person_id))

        for id_to_remove in ids_to_merge:
            if id_to_remove == new_id:
                # Remove counts in favour of deleted id
                # 19: {19:200, 23:30, 24: 12}
                for i in ids_to_merge:
                    if (i != new_id) and (i in self.tracklet_votes[new_id]):
                        del self.tracklet_votes[new_id][i]
            else:
                if id_to_remove in self.tracklet_votes:
                    del self.tracklet_votes[id_to_remove]

        # print("Tracklet votes are now {}".format(self.tracklet_votes))

        self.reidentifier.merge(new_id, ids_to_merge)

        for id_to_remove in ids_to_merge:
            # Target person not merged with herself, it will be kept
            # in the 'known_persons'.
            if new_id == id_to_remove: continue

            source_person = self.get_person_with_id(id_to_remove)
            target_person = self.get_person_with_id(new_id)

            if target_person is None:
                logger.info("Cannot remove {}, probably already removed"
                            .format(id_to_remove))
            else:
                target_person.merge(source_person)
                self.known_persons.remove(source_person)



# class TrackletManager(object):
#     """Class to manage tracklets and re-identification"""
#     def __init__(self,
#                  dlib_landmarks_file_name="",
#                  use_cuda=1,
#                  keep_threshold=0.1,
#                  reid_threshold=0.4,
#                  max_nb_features_per_id=200):
#         """
#         """
#         logger.info(self.__class__.__name__)

#         # Reidentification members
#         self.openface_computer = OpenfaceComputer(dlib_landmarks_file_name,
#                                                   useCuda=use_cuda)

#         self.reidentifier = Reidentifier(keep_threshold=keep_threshold,
#                                          reid_threshold=reid_threshold,
#                                          max_nb_features_per_id=max_nb_features_per_id)

#         ############################################################
#         # Reidentification part
#         ############################################################
#         # Minimum number of votes for an identity to:
#         #  - re-identify an non-identified tracklet
#         #  - merge to tracklets
#         # self.min_votes_for_reid = min_votes_for_reid

#         # Used when reidentification is performed
#         self.tracklet_id_to_person_id = {}

#         # Lists of tracklets that should not be merged together
#         # because obviously being of different persons.
#         #
#         # For instance, if mutually_exclusive is [[1,2,4], [3,2]]
#         # (which would mean that persons 1,2,4 have been seen together
#         # on the smae image, and later persons 2,3 also) and that for
#         # some reason (e.g. occlusion), the merge process wants to
#         # merge [1,2], then this is forbidden.
#         self.mutually_exclusive = set([])

#         # A dictionnary to accumulate votes for each tracklet. When
#         # the number of votes is larger than self.min_votes_for_reid
#         self.tracklet_votes = {}



#     def accumulate_votes(self, votes, person_id, tracklet_id):
#         """Accumulate votes to 'tracklet_votes' and take into account the
#            'mutually_exclusive' pairs

#         """
#         # Create if needed
#         if person_id not in self.tracklet_votes:
#             self.tracklet_votes[person_id] = {}

#         accumulated = self.tracklet_votes[person_id]

#         for k,v in votes.items():
#             p1 = (min(tracklet_id, k), max(tracklet_id, k))
#             p2 = (min(person_id,   k), max(person_id,   k))

#             # Don't accumulate when mutually exclusive
#             if (p1 not in self.mutually_exclusive) and \
#                (p2 not in self.mutually_exclusive):
#                 if k not in accumulated:
#                     accumulated[k] = 0
#                 accumulated[k] += v

#     def manage(self, image, bbs):
#         """Manage tracklets in a color image by re-assigning identities"""
#         # display = image.copy()

#         for bb in bbs:
#             tid = bb.pid

#             if tid not in self.tracklet_id_to_person_id:
#                 logger.info("Create person {}".format(tid))
#                 self.tracklet_id_to_person_id[tid] = tid

#             pid = self.tracklet_id_to_person_id[tid]

#             x0, x1, y0, y1 = scale_bounding_box(image, bb.y, bb.x,
#                                                 bb.h, bb.w, bb.H, bb.W)
#             x2, x3, y2, y3 = enlarge_bounding_box(x0, x1, y0, y1, image, 0.2)

#             crop = image[y2:y3,x2:x3]

#             features = self.openface_computer.compute_on_image(crop, 1)

#             if features.features is not None:
#                 self.reidentifier.add(features.features, bb.pid, crop)
#                 reid = self.reidentifier.identify(features.features)
#                 print("{} -> {}".format(bb.pid, reid.votes))
#                 self.accumulate_votes(reid.votes, pid, tid)
