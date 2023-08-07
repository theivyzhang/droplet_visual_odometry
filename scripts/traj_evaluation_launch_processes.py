#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 08-06-2023
# Purpose: a python file that concurrently runs visual odometry and ground truth extraction

# ROS node messages
print("extracting 1) visual odometry and 2) ground truth from rosbag for trajectory evaluation")

import sys
import os
import csv
import rosbag
import rospy

from sensor_msgs.msg import CompressedImage
from stag_ros.msg import StagMarkers

# other packages
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import sys
import yaml
from yaml.loader import SafeLoader
import transformations as tf

# import from modules
from visual_odometry_v3 import VisualOdometry  # visual odometry module
from traj_eval_ground_truth import GroundTruth  # ground truth reading module
import pose_estimation_module as PoseEstimationFunctions

import tf as tf


class Activate_GT_VO_Processes:
    def __init__(self, bag_file_path=None, gt_output_file_path='', vo_output_file_path=''):
        # set up flags
        self.previous_image = None
        self.current_image = None
        self.marker_reading = None
        self.first_topic_found = ""
        self.valid_count = 0

        # set up file paths
        self.bag_file_path = bag_file_path
        self.gt_output_file_path = gt_output_file_path
        self.vo_output_file_path = vo_output_file_path

        # clear existing data for sanity checks
        PoseEstimationFunctions.clear_txt_file_contents(self.gt_output_file_path)
        PoseEstimationFunctions.clear_txt_file_contents(self.vo_output_file_path)

        # initialize modules
        self.visual_odometry = VisualOdometry()  # default settings of visual odometry; visual odometry functions
        self.ground_truth = GroundTruth()

        # initialize lists
        self.vo_tf_list = []  # final list containing marker to marker transforms (ground truth)
        self.gt_tf_list = []  # final list containing marker to marker transforms (
        self.gt_camera_to_marker_list = []

        self.robot_starting_position_transformation = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.vo_tf_list.append(self.robot_starting_position_transformation)  # TODO: CHECK
        self.gt_tf_list.append(self.robot_starting_position_transformation)

        self.extract_vo_gt_data()

    def extract_vo_gt_data(self):
        with rosbag.Bag(self.bag_file_path, "r") as bag:
            gt_transformation = None
            for topic, bag_message, timestamp in bag.read_messages(topics=['/camera_array/cam0/image_raw/compressed',
                                                                           '/bluerov_controller/ar_tag_detector'],
                                                                   ):
                # print("{} {}".format(topic, timestamp.to_sec()))

                # print("topic found is {} at timestamp {}".format(topic, timestamp))
                if self.first_topic_found:

                    if topic != self.first_topic_found:
                        # print("different topic and topic is {}".format(topic))
                        # valid

                        # Saving ground truth
                        if gt_transformation is None and "ar_tag" in topic:
                            # tmp_gt_tf = bag_message
                            self.marker_reading = bag_message
                            # print("trying to see and len of marker is {}".format(len(self.marker_reading.markers)))
                            # print("marker reading message {}".format(self.marker_reading))

                            if len(self.marker_reading.markers) > 0:
                                # get the camera to marker transformation
                                gt_transformation = self.ground_truth.get_ground_truth_estimate(
                                    marker_reading=self.marker_reading)
                                # print("here is gt transform {}".format(gt_transformation))
                                # separate the translation and the quaternion
                                # TODO have the message already in the form of a transformation
                                self.gt_camera_to_marker_list.append(gt_transformation)

                                # print("list of ground truth transforms: {} of length {}".format(self.gt_tf_list, len(self.gt_tf_list)))

                                self.valid_count += 1

                            # TODO: get mTm transform between the last two cTm transforms; open ground truth txt and append data

                        gt_transformation = None  # Reset.
                        # self.valid_count += 1

                        if self.valid_count > 1:
                            if self.current_image is None and "image" in topic:
                                self.current_image = bag_message
                            # VO
                            current_image = self.visual_odometry.rosframe_to_current_image(self.current_image)
                            previous_image = self.visual_odometry.rosframe_to_current_image(self.previous_image)

                            # outputs transformation with respect to the current position of robot (different from default)
                            vo_transformation = self.visual_odometry.visual_odometry_calculations(current_image,
                                                                                                  previous_image,
                                                                                                  self.vo_tf_list[-1])
                            # print("visual odometry estimation: {}".format(vo_transformation))
                            self.vo_tf_list.append(vo_transformation)

                            # print("list of vo transforms {} of length {}".format(self.vo_tf_list, len(self.vo_tf_list)))

                            # extract translation and quaternion separately from the vo_transformation
                            vo_translation = PoseEstimationFunctions.translation_from_transformation_matrix(
                                transformation_matrix=vo_transformation)
                            vo_translation_unit = gt_translation / np.linalg.norm(
                                vo_translation)  # turn into unit vector
                            vo_quaternion = PoseEstimationFunctions.quaternion_from_transformation_matrix(
                                transformation_matrix=vo_transformation)
                            # print("estimated translation is {} and quaternion is {}".format(vo_translation,
                            #                                                               vo_quaternion))

                            # TODO: CHECK - write the data for visual odometry
                            PoseEstimationFunctions.write_to_output_file(self.vo_output_file_path, timestamp.to_sec(),
                                                                         vo_translation_unit, vo_quaternion)

                            # now get the marker_to_marker transform between previous and current
                            gt_mTm = PoseEstimationFunctions.get_marker_to_marker_transformation(
                                previous_cTm_transform=self.gt_tf_list[-1],
                                current_cTm_transform=self.gt_camera_to_marker_list[-1])

                            self.gt_tf_list.append(gt_mTm)
                            # print("the list of ground truth transforms {} with length {}".format(self.gt_tf_list, len(self.gt_tf_list)))

                            # separate the translation and the quaternion
                            gt_translation = PoseEstimationFunctions.translation_from_transformation_matrix(
                                transformation_matrix=gt_mTm)
                            gt_translation_unit = gt_translation / np.linalg.norm(
                                gt_translation)  # turn into unit vector
                            gt_quaternion = PoseEstimationFunctions.quaternion_from_transformation_matrix(
                                transformation_matrix=gt_mTm)
                            # print("ground truth translation is {} and quaternion is {}".format(gt_translation_unit,
                            #                                                               gt_quaternion))
                            # TODO: CHECK - write the data
                            PoseEstimationFunctions.write_to_output_file(self.gt_output_file_path, timestamp.to_sec(),
                                                                         gt_translation_unit, gt_quaternion)

                            # TODO: another module that does the calculations to have the same global reference frame ***

                            self.previous_image = self.current_image
                            self.current_image = None
                            self.valid_count = 1

                        # valid count <= 1
                        else:
                            if "image" in topic:
                                self.previous_image = bag_message
                                # print("adding initial position")
                                # vo_tf = np.eye(4, dtype=float)  # TODO ensure consistency with the other vo_tf; CHECKED
                                # self.vo_tf_list.append(vo_tf)

                        self.first_topic_found = ""

                    # if the same topic with first topic found
                    else:
                        self.first_topic_found = topic
                        if "image" in topic:
                            if self.valid_count < 1:
                                self.previous_image = bag_message
                            elif self.valid_count == 1:
                                self.current_image = bag_message
                        else:
                            self.marker_reading = bag_message

                else:
                    self.first_topic_found = topic
                    if "image" in topic:
                        if self.valid_count < 1:
                            self.previous_image = bag_message
                        elif self.valid_count == 1:
                            # print("now setting current image")
                            self.current_image = bag_message
                        else:
                            self.current_image = bag_message
                # print("valid count check {}".format(self.valid_count))

            print("finished bag!")


def main(args):
    rospy.init_node('LaunchProcessNode', anonymous=True)
    bag_file_path = '/home/ivyz/Documents/8-31-system-trials_2021-07-28-16-33-22.bag'
    gt_output_file_path = '/src/vis_odom/scripts/trajectory_evaluation/traj_eval_set1_08072023/stamped_ground_truth.txt'
    vo_output_file_path = '/src/vis_odom/scripts/trajectory_evaluation/traj_eval_set1_08072023/stamped_traj_estimate.txt'
    Activate_GT_VO_Processes(bag_file_path=bag_file_path, gt_output_file_path=gt_output_file_path,
                             vo_output_file_path=vo_output_file_path)

    print("trajectory evaluation processes activated")
    rospy.sleep(1)


# create the name function
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
