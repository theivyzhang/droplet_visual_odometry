#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 7-25-2023
# Purpose: An updated version for unit testing
# Pipeline: imports FrameExtraction to get real-time images, store as global variables and run detailed analysis

# ROS node messages
import rospy

# other packages
import cv2 as cv
import numpy as np
import sys
from visual_odometry_v2 import VisualOdometry


class UnitTestingComparing:
    def __init__(self):
        self.frame_translations = []
        # self.parse_camera_intrinsics()
        self.matches_dictionary = []
        self.robot_position_list = []
        self.previous_image = cv.imread("/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/unit_testing_07262023/test_set4_frame1.jpg")
        self.current_image = cv.imread("/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/unit_testing_07262023/test_set4_frame2.jpg")
        self.vo = VisualOdometry()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.orb_feature_detector = cv.ORB_create()
        # self.robot_curr_position = self.make_transform_mat(translation=[0, 0, 0], euler=[0, 0, 0])

        # self.previous_key_points = None  # same key points of PREVIOUS frame
        # self.previous_descriptors = None

        self.traj_estimates_file_path = "/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/stamped_traj_estimates.txt"


    def get_features_transformation(self):
        self.vo.visual_odometry_calculations(self.previous_image, None)
        self.vo.visual_odometry_calculations(self.current_image, self.previous_image)
        print("robot current position with previous AND current image: {}".format(self.vo.robot_curr_position))



        print("robot translated in visual odometry",self.vo.robot_current_translation)



def main(args):
    # rospy.init_node('VisualOdometryNode', anonymous=True)
    unit_testing_comparing = UnitTestingComparing()
    print("visual odometry activated")
    # rospy.sleep(1)
    unit_testing_comparing.get_features_transformation()


# create the name function
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass

