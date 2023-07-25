#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 7-25-2023
# Purpose: An updated version for unit testing
# Pipeline: imports FrameExtraction to get real-time images, store as global variables and run detailed analysis

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

from visual_odometry_v2 import VisualOdometry


class UnitTestingComparing:
    def __init__(self):
        self.frame_translations = []
        # self.parse_camera_intrinsics()
        self.matches_dictionary = []
        self.robot_position_list = []
        self.previous_image = cv.imread("/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/unit_testing_07242023/set4_ut_072423_frame1.jpg")
        self.current_image = cv.imread("/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/unit_testing_07242023/set4_ut_072423_frame2.jpg")
        self.vo = VisualOdometry()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.orb_feature_detector = cv.ORB_create()
        # self.robot_curr_position = self.make_transform_mat(translation=[0, 0, 0], euler=[0, 0, 0])

        # self.previous_key_points = None  # same key points of PREVIOUS frame
        # self.previous_descriptors = None

    def get_features_transformation(self):
        self.vo.visual_odometry_calculations(self.previous_image, None)
        self.vo.visual_odometry_calculations(self.current_image, self.previous_image)
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

