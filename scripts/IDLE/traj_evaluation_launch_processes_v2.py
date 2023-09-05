#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 08-08-2023
# Purpose: a TEST python file that concurrently runs visual odometry and ground truth extraction
# testing differences in handling matrix algebra

# ROS node messages
print("extracting 1) visual odometry and 2) ground truth from rosbag for trajectory evaluation")

import rosbag
import rospy

# other packages
import numpy as np
import sys

# import from modules
from visual_odometry_v3 import VisualOdometry  # visual odometry module
from traj_eval_ground_truth import GroundTruth  # ground truth reading module
import pose_estimation_module as PoseEstimationFunctions

class Activate_GT_VO_Processes:
    def __init__(self, bag_file_path=None, gt_output_file_path='', vo_output_file_path='', matching_mode='orb'):
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
        self.visual_odometry = VisualOdometry(to_sort=False, mode=matching_mode)
        print("activating mode {}".format(self.visual_odometry.mode))
        self.ground_truth = GroundTruth()

        # initialize lists
        self.vo_tf_list = []
        self.gt_tf_list = []
        self.gt_camera_to_marker_list = []
        self.gt_camera_to_camera_list = []

        # add starting positions
        self.robot_starting_position_transformation = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.vo_tf_list.append(self.robot_starting_position_transformation)
        self.gt_camera_to_camera_list.append(self.robot_starting_position_transformation)

        # start getting vo gt data
        self.extract_vo_gt_data()

    def extract_vo_gt_data(self):
        with rosbag.Bag(self.bag_file_path, "r") as bag:
            gt_transformation = None
            i = 0
            j  = 0
            for topic, bag_message, timestamp in bag.read_messages(topics=['/camera_array/cam0/image_raw/compressed',
                                                                           '/bluerov_controller/ar_tag_detector'],
                                                                   ):
                i += 1
                print("{} {} {}".format(i, topic, timestamp.to_sec()))

                # print("topic found is {} at timestamp {}".format(topic, timestamp))

                if not self.first_topic_found:
                    # finding first message for a valid pair
                    self.first_topic_found = topic
                    if "image" in topic:
                        # Saving image for when a valid pair is found
                        if self.valid_count < 1:
                            self.previous_image = bag_message
                        elif self.valid_count == 1:
                            self.current_image = bag_message
                        else:
                            self.current_image = bag_message
                    else:
                        # otherwise saving ground truth for when a valid pair is found.

                        # TODO avoid repetition of code
                        self.marker_reading = bag_message

                        if len(self.marker_reading.markers) > 0:
                            # get the camera to marker transformation
                            gt_transformation = self.ground_truth.get_ground_truth_estimate(
                                marker_reading=self.marker_reading)
                            # TODO try to include all markers -->

                            # separate the translation and the quaternion
                            if gt_transformation is not None:
                                # TODO have the message already in the form of a transformation - CHECKED
                                self.gt_camera_to_marker_list.append(gt_transformation)
                                # self.valid_count += 1 # TODO is valid_count needed or just len(gt_camera_to_marker_list is enough?

                else:
                    # found second message after finding a first message, searching for valid pair
                    if topic != self.first_topic_found:
                        # found actual valid pair image-pose or pose-image
                        # Saving ground truth
                        if gt_transformation is None and "ar_tag" in topic:
                            # tmp_gt_tf = bag_message
                            self.marker_reading = bag_message

                            if len(self.marker_reading.markers) > 0:
                                # get the camera to marker transformation
                                gt_transformation = self.ground_truth.get_ground_truth_estimate(
                                    marker_reading=self.marker_reading)
                                # print("here is gt transform {}".format(gt_transformation))
                                # separate the translation and the quaternion
                                # TODO have the message already in the form of a transformation
                                if gt_transformation is not None:
                                    self.gt_camera_to_marker_list.append(gt_transformation)

                                    self.valid_count += 1

                            # TODO: get mTm transform between the last two cTm transforms; open ground truth txt and append data


                        # self.valid_count += 1

                        if self.valid_count > 1 and gt_transformation is not None:
                            # Have at least 2 images with corresponding "ground truth", calculate VO


                            if self.current_image is None: #and "image" in topic:
                                self.current_image = bag_message
                            # TODO: CHECK - compute the data for visual odometry
                            current_image = self.visual_odometry.rosframe_to_current_image(self.current_image)
                            previous_image = self.visual_odometry.rosframe_to_current_image(self.previous_image)

                            # outputs transformation with respect to the current position of robot (different from default)
                            vo_transformation = self.visual_odometry.visual_odometry_calculations(previous_image,
                                                                                                  current_image,
                                                                                                  self.vo_tf_list[-1])
                            self.vo_tf_list.append(vo_transformation)

                            # extract translation and quaternion separately from the vo_transformation
                            vo_translation = PoseEstimationFunctions.translation_from_transformation_matrix(
                                transformation_matrix=vo_transformation)
                            vo_quaternion = PoseEstimationFunctions.quaternion_from_transformation_matrix(
                                transformation_matrix=vo_transformation)

                            PoseEstimationFunctions.write_to_output_file(self.vo_output_file_path, timestamp.to_sec(),
                                                                         vo_translation, vo_quaternion)


                            # TODO: CHECK - compute the data for ground truth; DIFFERENCE: getting camera_to_camera
                            # now get the marker_to_marker transform between previous and current
                            gt_cTc_length = len(self.gt_camera_to_camera_list)

                            # current cTc = gt_cTm[-2] dot inv(gt_cTm[-1]) dot previous cTc
                            # cTc at time 0 is the identity matrix
                            current_camera_to_camera_transform = np.matmul(self.gt_camera_to_camera_list[gt_cTc_length-1],
                                                                           np.matmul(self.gt_camera_to_marker_list[-2],
                                                                                     np.linalg.inv(
                                                                                         self.gt_camera_to_marker_list[
                                                                                             -1]))
                                                                           )


                            self.gt_camera_to_camera_list.append(current_camera_to_camera_transform)

                            # separate the translation and the quaternion
                            gt_translation = PoseEstimationFunctions.translation_from_transformation_matrix(
                                transformation_matrix=current_camera_to_camera_transform)
                            #gt_translation_unit = gt_translation / np.linalg.norm(
                                #gt_translation)  # turn into unit vector
                            gt_quaternion = PoseEstimationFunctions.quaternion_from_transformation_matrix(
                                transformation_matrix=current_camera_to_camera_transform)
                            # TODO: CHECK - write the data
                            PoseEstimationFunctions.write_to_output_file(self.gt_output_file_path, timestamp.to_sec(),
                                                                         #gt_translation_unit,
                                                                         gt_translation,
                                                                         gt_quaternion)

                            # TODO: another module that does the calculations to have the same global reference frame ***



                            self.previous_image = self.current_image
                            self.current_image = None
                            self.valid_count = 1
                            j += 1
                            if j > 1000:
                                print("completed unit testing")
                                break

                        # valid count <= 1
                        else:
                            if "image" in topic:
                                self.previous_image = bag_message

                        gt_transformation = None  # Reset.
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

            print("finished bag!")


def main(args):
    rospy.init_node('LaunchProcessNode', anonymous=True)
    bag_file_path = '/home/ivyz/Documents/8-31-system-trials_2021-07-28-16-33-22.bag'
    gt_output_file_path = '/src/vis_odom/scripts/trajectory_evaluation/KNN_SIFT_08162023_200/stamped_ground_truth.txt'
    vo_output_file_path = '/src/vis_odom/scripts/trajectory_evaluation/KNN_SIFT_08162023_200/stamped_traj_estimate.txt'

    Activate_GT_VO_Processes(bag_file_path=bag_file_path, gt_output_file_path=gt_output_file_path,
                             vo_output_file_path=vo_output_file_path, matching_mode='flann')

    print("trajectory evaluation processes activated")
    rospy.sleep(1)


# create the name function
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
