#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 08-16-2023
# Purpose: a python file that runs visual odometry and ground truth extraction on a single pair of images saved

# ROS node messages
print("extracting 1) visual odometry and 2) ground truth from rosbag for trajectory evaluation")

import rosbag
import rospy

# other packages
import numpy as np
import sys
import cv2 as cv
import tf as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import the Axes3D module

# import from modules
from visual_odometry_v3 import VisualOdometry  # visual odometry module
from traj_eval_ground_truth import GroundTruth  # ground truth reading module
import pose_estimation_module as PoseEstimationFunctions
import get_valid_message_stream as StreamExtract
import collections

TransformationPair = collections.namedtuple('TransformationPair', ['timestamp', 'transformation'])


class UnitTestingExtractData:
    def __init__(self, bag_file_path=None, gt_marker_positions_file_path='', gt_relative_transformations_file_path='',
                 vo_relative_transformations_file_path='', vo_absolute_position_file_path='', folder_path=' ',
                 matching_mode='orb', calibration_file_path=' ',
                 controlled=False, marker_id=0, real_marker_length=0.0):

        # clear existing data for sanity checks
        # set up file paths and clear existing content
        self.folder_path = folder_path
        self.bag_file_path = bag_file_path
        self.gt_relative_transformations_file_path = gt_relative_transformations_file_path
        self.gt_marker_positions_file_path = gt_marker_positions_file_path
        self.vo_relative_transformations_file_path = vo_relative_transformations_file_path
        self.vo_absolute_position_file_path = vo_absolute_position_file_path
        self.clear_file_contents_all()

        # set up flags
        self.previous_image = None
        self.current_image = None
        self.matching_mode = matching_mode
        self.calibration_file_path = calibration_file_path
        self.controlled = controlled
        self.marker_id_reference = marker_id

        # TODO: ADDED MARKER LENGTH IN REAL LIFE MEASUREMENTS (METERS) FOR SCALING FACTOR
        # PREMISE: scaling factor is different every frame given that the pixel length of the markers is constantly changing
        self.real_marker_length = real_marker_length

        self.base_link_flag = False  # flag to indicate if there is a base_link ref frame

        # initialize lists
        self.vo_absolute_position_list = []
        self.gt_marker_positions = []
        self.gt_camera_to_camera_list = []
        self.vo_camera_to_camera_list = []

        # start getting vo gt data
        self.valid_message_stream = StreamExtract.get_valid_message_stream(self.bag_file_path)

        # call the functions
        # 1) compute all gt vo
        self.compute_all_gt_vo_comparison_list()
        # 2) compute to output txt
        self.log_all_lists()
        # 3) plot all the graphs

    def clear_file_contents_all(self):
        # clear existing data for sanity checks
        PoseEstimationFunctions.clear_txt_file_contents(self.gt_relative_transformations_file_path)
        PoseEstimationFunctions.clear_txt_file_contents(self.gt_marker_positions_file_path)
        PoseEstimationFunctions.clear_txt_file_contents(self.vo_absolute_position_file_path)
        PoseEstimationFunctions.clear_txt_file_contents(self.vo_relative_transformations_file_path)

    def initialize_visual_odometry_module(self, robot_starting_position):  # initialize modules
        # Extract translation vector (first three elements of the last column)
        robot_starting_translation = robot_starting_position[:3, 3]

        # Extract rotation matrix (upper-left 3x3 submatrix)
        robot_starting_rotation = robot_starting_position[:3, :3]

        # Calculate Euler angles from the rotation matrix
        robot_starting_euler = tf.transformations.euler_from_matrix(robot_starting_rotation)
        self.visual_odometry = VisualOdometry(starting_translation=robot_starting_translation,
                                              starting_euler=robot_starting_euler, to_sort=False,
                                              mode=self.matching_mode,
                                              calibration_file_path=self.calibration_file_path,
                                              controlled=self.controlled,
                                              real_marker_length= self.real_marker_length)
        print("activating mode {}".format(self.visual_odometry.mode))

    def initialize_ground_truth_module(self):
        self.ground_truth = GroundTruth()

    def extract_and_compute_gt_marker_position(self, valid_set):
        # print(valid_set.marker_msg)
        # print("markers",valid_set.marker_msg.markers.markers)
        camera_T_marker_position = self.ground_truth.get_marker_position(
            marker_reading=valid_set.marker_msg,
            reference_id=self.marker_id_reference, base_link_flag=self.base_link_flag
        )
        print("receiving ground truth translation {}".format(camera_T_marker_position[:3, 3]))
        return camera_T_marker_position

    def extract_and_compute_gt_transformation(self):
        camera_T_camera_ground_truth = np.matmul(self.gt_marker_positions[-1].transformation,
                                                 np.linalg.inv(self.gt_marker_positions[-2].transformation))
        return camera_T_camera_ground_truth

    def get_ground_truth_marker_pixel_length(self, valid_set):
        current_marker_message = valid_set.marker_msg
        marker_pixel_length = self.ground_truth.get_current_marker_pixel_length(marker_message = current_marker_message)
        return marker_pixel_length

    def extract_and_compute_vo_robot_current_position(self):
        robot_current_position = self.visual_odometry.robot_curr_position
        return robot_current_position

    def extract_and_compute_vo_transformation(self, previous_valid_set, current_valid_set, marker_pixel_length):
        # process the consecutive image frames

        previous_image = self.visual_odometry.ros_img_msg_to_opencv_image(
            image_message=previous_valid_set.img_msg, msg_type='usb_raw')
        current_image = self.visual_odometry.ros_img_msg_to_opencv_image(
            image_message=current_valid_set.img_msg, msg_type='usb_raw')
        robot_current_position, robot_camera_to_camera = self.visual_odometry.visual_odometry_calculations(
            previous_image=previous_image,
            current_image=current_image,
            robot_previous_position_transformation=self.vo_absolute_position_list[-1].transformation,
            marker_pixel_length=marker_pixel_length)

        return robot_current_position, robot_camera_to_camera

    """Main functions"""

    def compute_all_gt_vo_comparison_list(self):
        previous_valid_set = None
        for i in range(0, len(self.valid_message_stream)):
            if previous_valid_set is None:
                previous_valid_set = self.valid_message_stream[i]
                timestamp = previous_valid_set.timestamp

                # initialize ground truth
                self.initialize_ground_truth_module()

                gt_marker_position = self.extract_and_compute_gt_marker_position(valid_set=previous_valid_set)
                self.gt_marker_positions.append(
                    TransformationPair(timestamp=timestamp, transformation=gt_marker_position))

                # TODO: activate VO here? passing marker position # 1 as robot starting position
                self.initialize_visual_odometry_module(
                    robot_starting_position=self.gt_marker_positions[0].transformation)

                # use first position of marker reading as starting position of the robot
                self.vo_absolute_position_list.append(
                    TransformationPair(timestamp=timestamp, transformation=self.visual_odometry.robot_curr_position))

                # activate visual odometry

            else:  # we have the first pair and need the second pair
                print("we have pair one and pair two")
                current_valid_set = self.valid_message_stream[i]
                timestamp = current_valid_set.timestamp

                # TODO - get marker position = absolute and relative ground truth position
                gt_marker_position = self.extract_and_compute_gt_marker_position(valid_set=current_valid_set)
                self.gt_marker_positions.append(
                    TransformationPair(timestamp=timestamp, transformation=gt_marker_position))

                marker_pixel_length = self.get_ground_truth_marker_pixel_length(valid_set=current_valid_set)

                gt_relative_position_change = self.extract_and_compute_gt_transformation()
                self.gt_camera_to_camera_list.append(
                    TransformationPair(timestamp=timestamp, transformation=gt_relative_position_change))

                # TODO - get and compute both robot absolute postion as well as transformation between frames
                robot_current_position, robot_camera_to_camera = self.extract_and_compute_vo_transformation(
                    previous_valid_set=previous_valid_set,
                    current_valid_set=current_valid_set,
                    marker_pixel_length=marker_pixel_length
                )
                self.vo_absolute_position_list.append(
                    TransformationPair(timestamp=timestamp, transformation=robot_current_position))
                self.vo_camera_to_camera_list.append(
                    TransformationPair(timestamp=timestamp, transformation=robot_camera_to_camera))

                previous_valid_set = current_valid_set

    def log_all_lists(self):
        # log marker positions (absolute ground truth)
        self.log_list_items_to_files(self.gt_marker_positions, False, self.gt_marker_positions_file_path)
        # log robot positions for robot (absolute vis odom)
        self.log_list_items_to_files(self.vo_absolute_position_list, True, self.vo_absolute_position_file_path)
        # log marker camera to camera transform (relative ground truth)
        self.log_list_items_to_files(self.gt_camera_to_camera_list, False, self.gt_relative_transformations_file_path)
        # log robot camera to camera transform (relative vis odom)
        self.log_list_items_to_files(self.vo_camera_to_camera_list, True, self.vo_relative_transformations_file_path)

    def extract_translation_quaternion_from_transform(self, transform, isVisOdom):
        translation = PoseEstimationFunctions.translation_from_transformation_matrix(transform)
        quaternion = PoseEstimationFunctions.quaternion_from_transformation_matrix(transform)
        return translation, quaternion

    def log_list_items_to_files(self, list_of_transforms, isVisOdom, output_file_path):
        for transform_pair in list_of_transforms:
            timestamp = transform_pair.timestamp
            transform = transform_pair.transformation

            translation, quaternion = self.extract_translation_quaternion_from_transform(transform=transform,
                                                                                         isVisOdom=isVisOdom)
            print("extracting translation {}".format(translation))

            with open(output_file_path, 'a') as file:
                file.write(str(timestamp) + " " + str(translation[0]) + " " + str(
                    translation[1]) + " " + str(translation[2]) + " "
                           + str(quaternion[0]) + " " + str(quaternion[1]) + " " + str(
                    quaternion[2]) + " " + str(quaternion[3]) + " " + "\n")


def main(folder_path, matching_mode, controlled, id, real_marker_length):
    rospy.init_node('UnitTestingControlledExperiment', anonymous=True)
    print("starting controlled experiments")
    calibration_file_path = '/home/ivyz/Documents/ivy_workspace/src/new_usb_cam.yaml'
    if controlled == 'controlled':
        is_controlled = True
    else:
        is_controlled = False


    bag_file_path = folder_path + "/scaled_with_bundles_forward.bag"
    print("successfully located bag {}".format(bag_file_path))
    gt_relative_transformations_file_path = folder_path + "/stamped_ground_truth_relative.txt"
    gt_marker_positions_file_path = folder_path + "/stamped_ground_truth_absolute.txt"
    vo_relative_transformations_file_path = folder_path + "/stamped_traj_estimate_relative.txt"
    vo_absolute_position_file_path = folder_path + "/stamped_traj_estimate_absolute.txt"

    # log marker details
    marker_id = id
    real_marker_length = real_marker_length

    UnitTestingExtractData(bag_file_path=bag_file_path,
                           gt_relative_transformations_file_path=gt_relative_transformations_file_path,
                           gt_marker_positions_file_path=gt_marker_positions_file_path,
                           vo_relative_transformations_file_path=vo_relative_transformations_file_path,
                           vo_absolute_position_file_path=vo_absolute_position_file_path,
                           folder_path=folder_path, matching_mode=matching_mode,
                           calibration_file_path=calibration_file_path, controlled=is_controlled,
                           marker_id = marker_id, real_marker_length = real_marker_length)
    rospy.sleep(1)


# create the name function
if __name__ == '__main__':
    try:
        folder_path = sys.argv[1]
        matching_mode = sys.argv[2]
        controlled = sys.argv[3]
        id = int(sys.argv[4])
        real_marker_length = float(sys.argv[5])
        main(folder_path, matching_mode, controlled, id, real_marker_length)
    except rospy.ROSInterruptException:
        pass
