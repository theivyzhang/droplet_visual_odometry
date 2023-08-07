#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 8-04-2023
# Purpose: this is a looped ground truth extraction where the camera to marker translation and quaternion representation of rotation is extracted,
# then stored in a txt file in the correct format

# ROS node messages
print("extracting ground truth from two consecutive frames and giving us the marker to marker transformations")

import rospy
from sensor_msgs.msg import CompressedImage
from stag_ros.msg import StagMarkers
# from stag_ros.msg import StagMarkers

# other packages
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import sys
import yaml
from yaml.loader import SafeLoader

# from hypothesis2 import VisualOdometry as vo
import tf as tf

# GLOBAL VARIABLES
DEFAULT_BASE_LINK_TOPIC = '/base_link'
DEFAULT_CAMERA_TOPIC = '/cam_0_optical_frame'


class GroundTruth:
    def __init__(self, default_base_link_topic=DEFAULT_BASE_LINK_TOPIC, default_camera_topic=DEFAULT_CAMERA_TOPIC):

        # print("quaternion matrix of homogenous transformations matrix [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0,0,1]] is {}".
        #       format(self.quaternion_representation(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0,0,1]]))))
        """
        by initializing vo, the image subscriber in hypothesis is activated
        uncomment the draw matches section that save the matched image
        """

        self.ground_truth_full_list_in_base_link = []
        self.ground_truth_list_cam_to_marker = []

        # set up the needed flags
        self.default_base_link_topic = default_base_link_topic
        self.default_camera_topic = default_camera_topic

        self.listener = tf.TransformListener()

    """
    this method returns the quaternion representation of a rotation matrix
    """

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
        quaternion_rep = self.rotation_matrix_to_quaternion(rotation_matrix)
        # print("here is the quaternion representation: {}".format(quaternion_rep))
        return quaternion_rep

    """
    these methods computes the camera to marker translation (ground truth) at a given frame
    """

    def get_base_to_marker_homogenous_transformation(self, marker):
        bTm_translation = marker.pose.pose.position
        btm_orientation = marker.pose.pose.orientation

        bTm_translation_array = np.array([bTm_translation.x, bTm_translation.y, bTm_translation.z])
        bTm_rotation_array = np.array([btm_orientation.x, btm_orientation.y, btm_orientation.z, btm_orientation.w])

        bTm_translation_mat = tf.transformations.translation_matrix(bTm_translation_array)
        bTm_orientation_mat = tf.transformations.quaternion_matrix(bTm_rotation_array)

        btm_homogenous_translation_mat = tf.transformations.concatenate_matrices(bTm_translation_mat,
                                                                                 bTm_orientation_mat)
        return btm_homogenous_translation_mat

    def get_camera_to_base_homogenous_transformation_matrix(self, marker):
        cTb_translation, cTb_rotation = self.listener.lookupTransform(self.default_base_link_topic,
                                                                      self.default_camera_topic,
                                                                      marker.header.stamp)
        # print("we have cTb translation {} and orientation {}".format(cTb_translation, cTb_rotation))

        cTb_translation_mat = tf.transformations.translation_matrix(cTb_translation)
        cTb_orientation_mat = tf.transformations.quaternion_matrix(cTb_rotation)
        # print("we have cTb translation matrix {} and orientation matrix {}".format(cTb_translation_mat,
        #                                                                            cTb_orientation_mat))

        cTb_homogenous_translation_mat = tf.transformations.concatenate_matrices(cTb_translation_mat,
                                                                                 cTb_orientation_mat)
        return cTb_homogenous_translation_mat

    def compute_frame_camera_to_marker(self, marker):
        # TODO:
        # for each frame, first get the pose translation and rotation info from base link
        # use tf transformations to get translation and rotation matrices
        # concatenate the two; the output is a homogenous transformation matrix 4x4, and yon now have base to marker
        btm_homogenous_translation_mat = self.get_base_to_marker_homogenous_transformation(marker)
        # print("we have the base to marker homogenous translation matrix {}".format(btm_homogenous_translation_mat))

        # PART B:
        # to get camera to baselink:
        # lookUpTransform produces a cam-2-baselink translation + rotation; repeat steps 2 - 3 in part A; produces 4x4 CTB
        # dot product: CTB dot BTM, you get CTM which is what the output of this function should be
        cTb_homogenous_translation_mat = self.get_camera_to_base_homogenous_transformation_matrix(marker)

        # print("we have the camera to base homogenous translation matrix {}".format(cTb_homogenous_translation_mat))

        # now get camera to marker transformation (4x4 homogenous transformation matrix)
        cam_to_marker_transformation = np.matmul(btm_homogenous_translation_mat, cTb_homogenous_translation_mat)

        # print("Here is the camera to marker transformation: {}".format(cam_to_marker_transformation))
        self.ground_truth_list_cam_to_marker.append(cam_to_marker_transformation)

        return cam_to_marker_transformation

    """
    this method gets the marker to marker translation every two consecutive frames with marker readings
    """

    def get_translation_between_two_frames(self, frame1_cTm, frame2_cTm):
        inverse_frame2_cTm = np.linalg.inv(frame2_cTm)
        marker_transform = np.matmul(inverse_frame2_cTm, frame1_cTm)
        # print("marker has translated {}".format(marker_transform))
        translation_only = np.array(
            [marker_transform.item(0, 3), marker_transform.item(1, 3), marker_transform.item(2, 3)])
        unit_translation = translation_only / np.linalg.norm(translation_only)

        ## make translation a unit vector ***

    def get_ground_truth_estimate(self, marker_reading):
        # callback function to access the ground truth data
        markers = marker_reading.markers  # get the marker information

        if len(markers) > 0:
            print("markers detected!")
            camera_to_marker_transformation = self.compute_frame_camera_to_marker(markers[0])
            return camera_to_marker_transformation
        # else:
        #     rospy.debug("No markers detected")



# create the name function
if __name__ == '__main__':
    pass
