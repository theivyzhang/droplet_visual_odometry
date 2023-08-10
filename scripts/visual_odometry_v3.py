#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 08-06-2023
# PURPOSE: A MODULE FOR VISUAL ODOMETRY POSE AND VELOCITY ESTIMATIONS FOR UNDERWATER AUTONOMOUS ROBOTS

#### ROS node messages
from geometry_msgs.msg import PoseStamped

#### other packages
import numpy as np
import cv2 as cv
import roslib
import os as os
import transformations as transf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from yaml.loader import SafeLoader

VERBOSE = False

number_of_frames = 25075

DEFAULT_STARTING_ROBOT_TRANSLATION = [0, 0, 0]
DEFAULT_STARTING_ROBOT_EULER = [0, 0, 0]


### UTILITY FUNCTION FOR TRANSFORMTION MATRIX ###

# TODO cleanup

class VisualOdometry:
    # Assumes that message is compressed, thus requiring frame width and height
    # global all_frames, previous_image, previous_key_points, previous_descriptors, current_frame, robot_position_list
    # TODO maybe not needed? parameters?
    frame_height = 1080
    frame_width = 1400

    def __init__(self, starting_translation=None, starting_euler=None, to_sort = False):
        if starting_euler is None:
            starting_euler = DEFAULT_STARTING_ROBOT_EULER
        if starting_translation is None:
            starting_translation = DEFAULT_STARTING_ROBOT_TRANSLATION
        self.starting_translation = starting_translation
        self.starting_euler = starting_euler
        self.robot_current_translation = None
        self.essential_matrix = None
        self.parse_camera_intrinsics()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # parameters needed for evaluating the image frames
        self.orb_feature_detector = cv.ORB_create()

        self.to_sort = to_sort

        self.matches_dictionary = []  # list of dictionaries

        # list to store the relevant positions
        self.robot_position_list = []  # calculated robot position list from opencv
        self.ground_truth_list = []  # list of ground truths from the ros node
        self.frame_translations = []  # maintains the translation between every consecutive frame
        self.marker_differences = []  # maintains the difference between the marker positions between every consecutive frames

        self.prev_transformation_matrix = self.make_transform_mat(translation=self.starting_translation,
                                                                  euler=self.starting_euler)
        self.robot_curr_position = self.make_transform_mat(translation=self.starting_translation,
                                                           euler=self.starting_euler)

        self.detected_marker = False

    """
    THIS IS THE SECTION CONTAINING ALL THE UTILITY FUNCTIONS
    """

    def undistort_image(self, distorted_image, new_camera_matrix):
        current_image = cv.undistort(distorted_image, self.int_coeff_mtx, self.dist_coef_arr, None, new_camera_matrix)
        return current_image

    def rosframe_to_current_image(self, frame):
        # compute the camera matrix
        new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(
            self.int_coeff_mtx,
            self.dist_coef_arr,
            (self.frame_width, self.frame_height),
            1,
            (self.frame_width, self.frame_height)
        )
        # convert the frame data into a numpy array
        np_arr = np.fromstring(frame.data, np.uint8)
        image_np = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        grey_image = cv.cvtColor(image_np, cv.COLOR_BGR2GRAY)
        current_image = self.undistort_image(grey_image, new_camera_matrix)
        return current_image

    def make_transform_mat(self, translation, euler):
        rx, ry, rz = euler
        rotation = transf.euler_matrix(rx, ry, rz, axes='sxyz')
        translation = transf.translation_matrix(translation)
        return translation.dot(rotation)

    def parse_camera_intrinsics(self):
        calibration_file_path = '/home/ivyz/Documents/ivy_workspace/src/vis_odom/Parameters/camera_calibration.yaml'
        with open(calibration_file_path) as camera_calibration:
            data = yaml.load(camera_calibration, Loader=SafeLoader)

        # input the distortion coefficients into an array
        distortion_coefficients = data["distortion_coeffs"]
        self.dist_coef_arr = distortion_coefficients[0]
        self.dist_coef_arr = np.array(self.dist_coef_arr).reshape(1, 5)

        # compute the 3x3 intrinsic coefficient matrix
        intrinsic_coefficients = data["intrinsic_coeffs"]
        self.int_coef_arr = intrinsic_coefficients[0]

        self.int_coeff_mtx = np.array(self.int_coef_arr)
        self.int_coeff_mtx = self.int_coeff_mtx.reshape(3, 3)

    """
    THIS SECTION CONTAINS ALL THE FUNCTIONS NEEDED FOR THE VISUAL ODOMETRY CALLBACK
    """

    def visualize_key_points_matching(self, current_descriptors, current_key_points,
                                      current_image_with_keypoints_drawn):

        # Match descriptors.
        matches = self.bf.match(previous_descriptors, current_descriptors)

        # TODO: discuss the possible ways to sort the matches
        # one way is using the ascending order of DMatch distances
        # the distance is apparently not the euclidean distances
        # key points: in what unit?
        # if keypoints are the coordinates of matches on the image, sorting by euclidean distance and/or slope between the points might be better

        # matches = sorted(matches, key=lambda x: x.distance)

        img3 = cv.drawMatches(self.previous_image, current_key_points, current_image_with_keypoints_drawn,
                              current_key_points, matches[:10], None,
                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        image_path = "/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/unit_testing_07242023/set5_ut_072423_matches.jpg"
        cv.imwrite(image_path, img3)

        cv.imshow('pattern', img3), cv.waitKey(5)

    def get_matches_between_two_frames(self, current_key_points, current_descriptors, previous_key_points,
                                       previous_descriptors, to_sort):

        top_ten_previous_key_points = []
        top_ten_current_key_points = []

        matches = self.bf.match(previous_descriptors, current_descriptors)
        if to_sort:
            matches = sorted(matches, key=lambda x: x.distance)
        # matches = sorted(matches, key=lambda x: x.distance)
        for i in range(len(matches[:10])):
            train_index = matches[i].trainIdx  # index of the match in previous key points
            query_index = matches[i].queryIdx  # index of the match in current key points
            top_ten_previous_key_points.append(previous_key_points[train_index])
            top_ten_current_key_points.append(current_key_points[query_index])

        return matches[:10], top_ten_previous_key_points, top_ten_current_key_points  # TODO parameter for top-k matches

    # TODO: CHECKED - PREV CURR
    def get_translation_between_two_frames(self, array_previous_key_points,
                                           array_current_key_points):  # TODO check for consistency previous-current

        # get the essential matrix
        # CHECK: camera intrinsic coefficient matrix check passed
        self.essential_matrix, mask = cv.findEssentialMat(points1=array_current_key_points,
                                                          points2=array_previous_key_points,
                                                          cameraMatrix=self.int_coeff_mtx,
                                                          method=cv.RANSAC, prob=0.999, threshold=1.0)

        # findEssentialMat expects the first two arguments to be numpy arrays

        # compute the relative position using the essential matrix, key points  using cv.relativepose
        points, relative_rotation, translation, mask = cv.recoverPose(E=self.essential_matrix,
                                                                      points1=array_current_key_points,
                                                                      points2=array_previous_key_points)
        translation = translation.transpose()[0]
        relative_rotation = np.array(relative_rotation)
        # decompose rotation matrix + find euler
        t = np.array([0, 0, 0])
        new_rotation_mat = np.vstack((np.hstack((relative_rotation, t[:, None])), [0, 0, 0, 1]))

        # get euler angles
        euler = transf.euler_from_matrix(new_rotation_mat, 'rxyz')

        # compute the current transformation matrix
        euler = np.array(euler)

        prev_to_curr_translation = self.make_transform_mat(translation=translation, euler=euler)

        # store previous to current translation in the corresponding list
        self.frame_translations.append(prev_to_curr_translation)
        return prev_to_curr_translation

    # here you would want to pass in the top 10 key points
    # TODO: CHECKED - PREV CURR
    def previous_current_matching(self, top_10_previous_key_points, top_10_current_key_points,
                                  robot_previous_position_transformation):

        """"you can choose to visualize the points matched between every two frames by uncommenting this line"""""

        # convert previous_key_points and current_key_points into floating point arrays
        array_previous_key_points = cv.KeyPoint_convert(top_10_previous_key_points)

        # top 10 previous key points is a list of KeyPoint objects, array_previous_key points is a numpy array
        array_current_key_points = cv.KeyPoint_convert(top_10_current_key_points)

        # get the translation between previous image and the current image using the array list forms of previous and current key points
        prev_to_curr_translation = self.get_translation_between_two_frames(array_previous_key_points,
                                                                           array_current_key_points)

        # calculate the current position using the previous-to-current translation
        robot_current_position_transformation = robot_previous_position_transformation.dot(prev_to_curr_translation)
        return robot_current_position_transformation

    def compute_current_image_elements(self, input_image):

        key_points = self.orb_feature_detector.detect(input_image, None)

        # find the keypoints and descriptors with ORB
        key_points, descriptors = self.orb_feature_detector.compute(input_image, key_points)
        # current key points output a list of KeyPoint elements, current descriptors is numpy array of (500 x 32 dimensions)
        image_with_keypoints_drawn = cv.drawKeypoints(input_image, key_points, None,
                                                      color=(0, 255, 0), flags=0)
        # current image with keypoints drawn is a numpy array

        return key_points, descriptors, image_with_keypoints_drawn

    def visual_odometry_calculations(self, previous_image, current_image, robot_previous_position_transformation):
        # get the relevant information needed for computing the translation
        previous_key_points, previous_descriptors, previous_image_with_keypoints_drawn = self.compute_current_image_elements(
            previous_image)

        current_key_points, current_descriptors, current_image_with_keypoints_drawn = self.compute_current_image_elements(
            current_image)

        # from the above code we produce a length 500 current key points and a (500 height, 32 width) shaped current descriptors
        top_ten_matches, top_10_previous_key_points, top_10_current_key_points = self.get_matches_between_two_frames(
            current_key_points, current_descriptors, previous_key_points, previous_descriptors, to_sort=self.to_sort)
        # get matches should return top 10 previous key points

        robot_current_position_transformation = self.previous_current_matching(top_10_previous_key_points,
                                                                               top_10_current_key_points,
                                                                               robot_previous_position_transformation)

        return robot_current_position_transformation  # TODO check if it is the 4x4 homogeneous matrix and rename properly to reflect that is the matrix - CHECKED

    def plot_graph(self, position_history, ax):

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        position_xs = [
            p[0] for p in position_history
        ]
        position_ys = [
            p[1] for p in position_history
        ]
        position_zs = [
            p[2] for p in position_history
        ]

        ax.plot(position_xs, position_ys, position_zs, c='orange')
        ax.scatter(position_xs, position_ys, position_zs, marker="x", c='cyan')

        print("finished plotting")

    def setup_plots(self):
        plt.ion()
        self.robot_pos_fig = plt.figure(figsize=(12, 8))
        self.ground_truth_fig = plt.figure(figsize=(12, 8))

        # settings for robot position list
        self.ax1 = self.robot_pos_fig.add_subplot(111, projection='3d')
        self.ax1.set_title("Visual odometry trajectory")
        self.ax1.set_xlabel('X Label')
        self.ax1.set_ylabel('Y Label')
        self.ax1.set_zlabel('Z Label')

        # settings for ground truth list
        self.ax2 = self.ground_truth_fig.add_subplot(111, projection='3d')
        self.ax2.set_title("From visual fiducials")
        self.ax2.set_xlabel('X Label')
        self.ax2.set_ylabel('Y Label')
        self.ax2.set_zlabel('Z Label')

        self.combined_fig = plt.figure(figsize=(12, 8))
        self.combined_ax = self.combined_fig.add_subplot(111, projection='3d')
        plt.show()

    def as_three_lists(self, positions):
        return [x[0] for x in positions], [x[1] for x in positions], [x[2] for x in positions]

    def plot_position_information(self):
        self.plot_graph(self.robot_position_list, self.ax1)
        print("plotting {} data points from robot position list".format(len(self.robot_position_list)))
        self.plot_graph(self.ground_truth_list, self.ax2)
        print("plotting {} data points from ground truth position list".format(len(self.ground_truth_list)))

        # the combined graph
        gt_x, gt_y, gt_z = self.as_three_lists(self.ground_truth_list)
        pred_x, pred_y, pred_z = self.as_three_lists(self.robot_position_list)
        self.combined_ax.plot(
            gt_x,
            gt_y,
            gt_z,
            c='lime',
            label='ground_truth'
        )
        self.combined_ax.plot(
            pred_x,
            pred_y,
            pred_z,
            c='orange',
            label='predicted'
        )
        self.combined_ax.set_title("Combined")
        plt.pause(0.1)


# create the name function
if __name__ == '__main__':
    pass
