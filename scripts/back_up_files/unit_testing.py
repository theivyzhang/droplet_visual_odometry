#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 7-10-2023
# Purpose: An updated version for unit testing
# Pipeline: imports FrameExtraction to get real-time images, write to txt file

# import visual odometry model --> change hypothesis to visual_odometry when finalized
from hypothesis import VisualOdometry
from frame_extraction import FrameExtraction

# ROS node messages
import rospy
from sensor_msgs.msg import CompressedImage
from stag_ros.msg import StagMarkers
from geometry_msgs.msg import PoseStamped

# other packages
import numpy as np
import cv2 as cv
import roslib
from cv_bridge import CvBridge, CvBridgeError
import sys
import os as os
import transformations as transf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from yaml.loader import SafeLoader

# GLOBAL VARIABLES
PREVIOUS_IMAGE = None
CURRENT_IMAGE = None


class UnitTest:
    def __init__(self):
        # first extract the images
        self.adjacent_image_extraction = FrameExtraction()

        self.bridge = CvBridge()

        # initialize visual odometry node
        self.vo_node = VisualOdometry()
        self.vo_node.parse_camera_intrinsics()

        self.prev_transformation_matrix = self.vo_node.make_transform_mat(translation=[0, 0, 0], euler=[0, 0, 0])
        print("Previous transform matrix at initialization: {}".format(self.prev_transformation_matrix))
        self.robot_curr_position = self.vo_node.make_transform_mat(translation=[0, 0, 0], euler=[0, 0, 0])
        print("robot_curr_position at initialization: {}".format(self.robot_curr_position))

        self.detected_marker = False
        self.start_feature_matching = False

        self.ground_truth_full_list = []

        self.previous_frame = self.adjacent_image_extraction.frame_one
        self.current_frame = self.adjacent_image_extraction.frame_two
        print("on initialization: are the frames the same? ", self.previous_frame==self.current_frame)


    """THIS PART OF THE CODE CONTAINS MISCELLANEOUS HELPER FUNCTIONS"""

    def show_two_frames(self):
        pass

    """THIS PART OF THE CODE COMPUTES THE GROUND TRUTH COORDINATES AND CALCULATES THE TRANSFORMATION"""

    def get_ground_truth_coordinates_of_group_of_frames(self, frames):
        for frame in frames:
            self.vo_node.marker_callback(msg=frame)
            self.ground_truth_full_list.append(self.vo_node.ground_truth_list)
        print("Length of ground truth full list: ", len(self.ground_truth_full_list))
        print("Ground truth full list: {}".format(self.ground_truth_full_list))

    def compute_ground_truth_of_one_single_frame(self):
        frames = [self.previous_frame]
        self.get_ground_truth_coordinates_of_group_of_frames(frames)

    def compute_ground_truth_of_two_frames(self):
        frames = [self.previous_frame, self.current_frame]
        self.get_ground_truth_coordinates_of_group_of_frames(frames)

    def are_they_the_same(self):  # note: only two frames to compare
        print("Are the two frames the same? {}".format(self.previous_frame == self.current_frame))
        print("Are the two frames the same format? {}, {}".format(type(self.previous_frame), type(self.current_frame)))

        cv.imshow("previous",self.previous_frame)
        cv.imshow("current", self.current_frame)



    """THIS PART OF THE CODE COMPUTES THE VISUAL ODOMETRY DATA AND CALCULATE THE PREDICTED TRANSFORMATION"""

    # def get_matches_and_transformations(self):
    #     previous_key_points, previous_descriptors, previous_image_with_keypoints_drawn = self.vo_node.compute_current_image_elements(
    #         self.vo_node.previous_image)
    #
    #     print("typeeee", type(previous_image_with_keypoints_drawn))
    #
    #
    #     current_key_points, current_descriptors, current_image_with_keypoints_drawn = self.vo_node.compute_current_image_elements(
    #         self.current_image)
    #     print("Current key points: {}".format(current_key_points))
    #     print("Current descriptors: {}".format(current_descriptors))
    #
    #     self.vo_node.previous_current_matching(previous_image_with_keypoints_drawn, current_key_points, current_descriptors,
    #                                            current_image_with_keypoints_drawn)
    # print("The essential matrix for image 0 and image 1: {}".format(self.vo_node.essential_matrix))
    # print("The essential matrix for image 0 and image 1: {}".format(self.vo_node.essential_matrix))


def main(args):
    rospy.init_node('VisualOdometryNode', anonymous=True)
    two_images = UnitTest()
    print("visual odometry activated")
    rospy.sleep(1)

    """UNCOMMENT the following line to compute ground truth of one single frame"""
    # two_images.compute_ground_truth_of_one_single_frame()

    """UNCOMMENT the following line to compute ground truth of two frames"""
    two_images.compute_ground_truth_of_two_frames()

    """UNCOMMENT the following line if you want to check if ground truths labelled are exactly the same"""
    print(two_images.are_they_the_same())
    # print(two_images.adjacent_image_extraction.are_they_the_same())

    # two_images.get_matches_and_transformations()


# create the name function
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
