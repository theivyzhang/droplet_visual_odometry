#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 7-27-2023
# Purpose: a python program to get visual odometry estimates from
# Pipeline: imports FrameExtraction to get real-time images, store as global variables and run detailed analysis

# ROS node messages
import rospy

# other packages
import cv2 as cv
import numpy as np
import transformations as tf
import sys
from visual_odometry_v2 import VisualOdometry

TRANSLATION_FROM_PREVIOUS_FRAME = []
EULER_FROM_PREVIOUS_FRAME = []

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
        self.traj_estimates_file_path = "/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/stamped_traj_estimates.txt"
        # self.robot_curr_position = self.make_transform_mat(translation=[0, 0, 0], euler=[0, 0, 0])

        # self.previous_key_points = None  # same key points of PREVIOUS frame


    def rotation_matrix_to_quaternion(self, rotation_matrix):
        trace = np.trace(rotation_matrix)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * S
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        elif (rotation_matrix[0, 0] > rotation_matrix[1, 1]) and (rotation_matrix[0, 0] > rotation_matrix[2, 2]):
            S = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2.0
            w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            x = 0.25 * S
            y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            S = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2.0
            w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            y = 0.25 * S
            z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
        else:
            S = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2.0
            w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
            x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
            y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
            z = 0.25 * S

        return [x, y, z, w]

    def quaternion_representation(self, homogenous_matrix):
        rotation_matrix = homogenous_matrix[:3, :3]
        # print("here is the rotation matrix: {}".format(rotation_matrix))
        quaternion_rep =self.rotation_matrix_to_quaternion(rotation_matrix)
        # print("here is the quaternion representation: {}".format(quaternion_rep))
        return quaternion_rep

    def get_features_transformation(self):
        self.vo.visual_odometry_calculations(self.previous_image, None)
        print("robot current position with only previous image: {}".format(self.vo.robot_curr_position))
        self.vo.visual_odometry_calculations(self.current_image, self.previous_image)
        print("robot current position with previous AND current image: {}".format(self.vo.robot_curr_position))
        print("robot translated in visual odometry",self.vo.robot_current_translation)

        robot_curr_pos_matrix = self.vo.robot_curr_position
        robot_curr_quaternion = self.quaternion_representation(robot_curr_pos_matrix)
        euler_angles = tf.euler_from_matrix(robot_curr_pos_matrix)
        print("here are the euler angles: {}".format(euler_angles))
        translation_vector = [robot_curr_pos_matrix[0, 3], robot_curr_pos_matrix[1, 3], robot_curr_pos_matrix[2, 3]]
        traj_estimate_line = str(translation_vector[0])+" "+str(translation_vector[1])+" "+str(translation_vector[2])+" "+\
                             str(robot_curr_quaternion[0])+" "+str(robot_curr_quaternion[1])+" "+str(robot_curr_quaternion[2])+" "+str(robot_curr_quaternion[3])

        # append the line to trajectory estimates txt file
        print(traj_estimate_line)
        try:
            with open(self.traj_estimates_file_path, 'a') as file:
                file.write(traj_estimate_line+"\n")

        except:
            print("cannot write in output file")


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

