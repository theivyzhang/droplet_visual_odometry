#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 08-04-2023
# Purpose: a python program to get visual odometry estimates from
# Pipeline: imports FrameExtraction to get real-time images, store as global variables and run detailed analysis

# ROS node messages
import rospy

# other packages
import cv2 as cv
import numpy as np
import transformations as tf
import sys
import os
from visual_odometry_v2 import VisualOdometry

TRANSLATION_FROM_PREVIOUS_FRAME = []
EULER_FROM_PREVIOUS_FRAME = []


class CollectImagePaths:
    def __init__(self, desktop_folder_path):
        self.output_image_paths = []
        self.desktop_folder_path = desktop_folder_path
        self.get_image_paths_from_folder(folder_path=self.desktop_folder_path)

    def get_image_paths_from_folder(self, folder_path):
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.jpg') or filename.endswith('.JPG'):
                image_path = os.path.join(folder_path, filename)
                self.output_image_paths.append(image_path)
        # print("output image paths: {}".format(self.output_image_paths))


class TrajectoryEvaluation_VisOdom_Collect_Data:
    def __init__(self, list_of_images_paths):
        print("trajectory estimation for visual odometry activated")
        self.list_of_images_paths = list_of_images_paths
        self.list_of_consecutive_cam2cam_transforms = []  # index 0 = cam0 T cam 1, index 1 = cam1 T cam2
        self.compute_transforms_and_write_data()

    def compute_transforms_and_write_data(self):
        # global  starting_translation, euler_angles
        transform_dict = {}
        for index in range(len(self.list_of_images_paths) - 1):
            current_image_path = self.list_of_images_paths[index + 1]
            previous_image_path = self.list_of_images_paths[index]
            # print("evaluating previous image: {}".format(previous_image_path))
            # print("evaluating current image: {}".format(current_image_path))
            if index == 0:
                starting_translation = [0, 0, 0]
                starting_euler = [0, 0, 0]
            else:
                previous_transform = self.list_of_consecutive_cam2cam_transforms[index - 1]
                # print("here is previous transform {} and its type {}".format(previous_transform,
                #                                                              type(previous_transform)))
                starting_translation = [previous_transform[0, 3], previous_transform[1, 3], previous_transform[2, 3]]
                euler_angles = tf.euler_from_matrix(previous_transform)
                starting_euler = [euler_angles[0], euler_angles[1], euler_angles[2]]
            # print("for set {}, we have the starting translation at {} and the starting euler at {}".format(index,
            #                                                                                                starting_translation,
            #                                                                                                starting_euler))
            trajestimate_visodom = TrajEstimates_VisualOdom_Evaluate_Data(previous_image_path=previous_image_path,
                                                                          current_image_path=current_image_path,
                                                                          current_starting_translation=starting_translation,
                                                                          current_starting_euler=starting_euler)
            current_cam2cam_transform = trajestimate_visodom.robot_curr_pos_matrix
            # print("here is the current cam to cam transform: {}".format(current_cam2cam_transform))
            self.list_of_consecutive_cam2cam_transforms.append(current_cam2cam_transform)
            dict_key = "cam" + str(index) + "to cam" + str(index + 1)
            transform_dict[dict_key] = current_cam2cam_transform
        print("now we have the full list of transforms: {}".format(self.list_of_consecutive_cam2cam_transforms))
        print("and the length of the transforms: {}".format(len(self.list_of_consecutive_cam2cam_transforms)))
        # sorted_transform_dict = {key : transform_dict[key] for key in sorted(transform_dict)}
        # print("now we also have the full dictionary of transforms: {}".format(sorted_transform_dict))
    # rospy.sleep(1)


class TrajEstimates_VisualOdom_Evaluate_Data:
    def __init__(self, previous_image_path, current_image_path, current_starting_translation, current_starting_euler):
        self.robot_curr_pos_matrix = None
        self.frame_translations = []
        # self.parse_camera_intrinsics()
        self.matches_dictionary = []
        self.robot_position_list = []
        self.previous_image = cv.imread(previous_image_path)
        self.current_image = cv.imread(current_image_path)
        self.vo = VisualOdometry(starting_translation=current_starting_translation,
                                 starting_euler=current_starting_euler)
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.orb_feature_detector = cv.ORB_create()
        self.traj_estimates_file_path = "/src/vis_odom/scripts/stamped_traj_estimate.txt"
        # self.robot_curr_position = self.make_transform_mat(translation=[0, 0, 0], euler=[0, 0, 0])

        # self.previous_key_points = None  # same key points of PREVIOUS frame
        self.get_features_transformation()

    def get_features_transformation(self):
        self.vo.visual_odometry_calculations(self.previous_image, None)
        # print("robot current position with only previous image: {}".format(self.vo.robot_curr_position))
        self.vo.visual_odometry_calculations(self.current_image, self.previous_image)
        # print("robot current position with previous AND current image: {}".format(self.vo.robot_curr_position))
        # print("robot translated in visual odometry", self.vo.robot_current_translation)
        self.robot_curr_pos_matrix = self.vo.robot_curr_position

class TrajectoryEvaluation_WriteData:
    def __init__(self, list_of_transforms, output_file_name):
        self.list_of_transforms = list_of_transforms
        self.file_name = output_file_name
        self.write_translation_and_quaternion_to_file()

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

    def write_translation_and_quaternion_to_file(self):
        rolling_transform = self.list_of_transforms[0]
        for i in range(1, len(self.list_of_transforms)):
            # get the translation
            current_translation = [rolling_transform[0, 3], rolling_transform[1, 3], rolling_transform[2, 3]]
            current_quaternion = self.quaternion_representation(rolling_transform)
            # write the information needed for the trajectory evaluation

            rolling_transform = rolling_transform * self.list_of_transforms[i]
            with open(self.file_name, 'a') as file:
                output_line = str(current_translation[0]) + " " + str(
                    current_translation[1]) + " " + str(current_translation[2]) + " " + str(
                    current_quaternion[0]) + " " + str(current_quaternion[1]) + " " + str(
                    current_quaternion[2]) + " " + str(current_quaternion[3]) + "\n"
                file.write(output_line)
                # print("writing: {}".format(output_line))


def main(args):
    # rospy.init_node('VisualOdometryNode', anonymous=True)
    desktop_folder_path = '/src/vis_odom/scripts/images/unit_testing_07282023_trial2'  # Replace with the actual folder path on your desktop\

    # first collect the images and extract the needed folders
    collect_images = CollectImagePaths(desktop_folder_path=desktop_folder_path)
    list_of_images_path = collect_images.output_image_paths[1:len(collect_images.output_image_paths) - 1]
    print("length of images path :{}".format(len(list_of_images_path)))
    # print("here are the images used for trajectory evaluation: {}".format(list_of_images_path))

    # the start the trajectory evaluation data collection
    collect_evaluate_data = TrajectoryEvaluation_VisOdom_Collect_Data(list_of_images_paths=list_of_images_path)
    list_of_cam2cam_transforms = collect_evaluate_data.list_of_consecutive_cam2cam_transforms

    # next we can start writing data for trajectory evaluation
    traj_estimate_file_path = "/src/vis_odom/scripts/stamped_traj_estimate.txt"
    TrajectoryEvaluation_WriteData(list_of_cam2cam_transforms, traj_estimate_file_path)
    print("trajectory estimation for visual odometry activated")
    rospy.sleep(1)


# create the name function
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
