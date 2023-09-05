#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 08-10-2023
# PURPOSE: A MODULE FOR VISUAL ODOMETRY POSE AND VELOCITY ESTIMATIONS FOR UNDERWATER AUTONOMOUS ROBOTS

# import needed packages
import numpy as np
import cv2 as cv
import transformations as transf
import yaml
from yaml.loader import SafeLoader
import pose_estimation_module as PEM


VERBOSE = False

number_of_frames = 25075

DEFAULT_STARTING_ROBOT_TRANSLATION = [0, 0, 0]
DEFAULT_STARTING_ROBOT_EULER = [0, 0, 0]


class VisualOdometry:
    # global all_frames, previous_image, previous_key_points, previous_descriptors, current_frame, robot_position_list

    def __init__(self, starting_translation=None, starting_euler=None, to_sort=False, mode="ORB",
                 calibration_file_path="", controlled=False):
        if starting_euler is None:
            starting_euler = DEFAULT_STARTING_ROBOT_EULER
        if starting_translation is None:
            starting_translation = DEFAULT_STARTING_ROBOT_TRANSLATION

        # sets frame width and height according to type (controlled = usb_cam/image_raw, otherwise compressed_image)
        self.controlled = controlled
        print("is this a controlled experiment? {}".format(self.controlled))
        if not controlled:
            self.frame_height = 1080
            self.frame_width = 1400
        else:
            self.frame_height = 480
            self.frame_width = 640

        print("have frame width {} and height {}".format(self.frame_width, self.frame_height))

        # the translation and euler at the start of the program
        self.starting_translation = starting_translation
        self.starting_euler = starting_euler

        # global variables
        self.robot_current_translation = None
        self.essential_matrix = None

        # parse the camera intrinsics before starting anything else
        self.calibration_file_path = calibration_file_path

        self.distortion_coefficient_matrix = None
        self.intrinsic_coefficient_matrix = None
        self.parse_camera_intrinsics()

        # model parameters
        self.to_sort = to_sort  # whether we want to sort the frames or not
        self.mode = mode

        # get the parameters needed under the specified mode
        self.feature_detector, self.norm_type, self.cross_check = self.return_feature_matching_parameters(mode)
        self.calibration_file_path = calibration_file_path

        # parameters needed for evaluating the image frames

        self.bf = cv.BFMatcher(normType=self.norm_type, crossCheck=self.cross_check)

        # list to store the relevant positions
        self.robot_position_list = []  # calculated robot position list from opencv
        self.ground_truth_list = []  # list of ground truths from the ros node
        self.frame_translations = []  # maintains the translation between every consecutive frame
        self.matches_dictionary = []  # list of dictionaries

        # initialize current position of the robot based on the translation and euler at the start of the program
        self.robot_curr_position = self.make_transform_mat(translation=self.starting_translation,
                                                           euler=self.starting_euler)


    """
    THIS IS THE SECTION CONTAINING ALL THE UTILITY FUNCTIONS
    """

    def return_feature_matching_parameters(self, mode):
        global feature_detector, norm_type, cross_check
        if mode.lower() == "orb":
            feature_detector = cv.ORB_create()
            norm_type = cv.NORM_HAMMING
            cross_check = True
        elif mode.lower() == "sift" or mode.lower() == 'flann' or mode.lower() == 'knn_sift':
            feature_detector = cv.xfeatures2d.SIFT_create()
            norm_type = cv.NORM_L1
            cross_check = False
        elif mode.lower() == 'surf':
            feature_detector = cv.xfeatures2d.SURF_create(400)
            norm_type = cv.NORM_L1
            cross_check = False
        return feature_detector, norm_type, cross_check

    # method to undistort the image
    def undistort_image(self, distorted_image, new_camera_matrix):
        current_image = cv.undistort(src=distorted_image, cameraMatrix=self.intrinsic_coefficient_matrix,
                                     distCoeffs=self.distortion_coefficient_matrix, newCameraMatrix=new_camera_matrix)
        return current_image

    def ros_img_msg_to_opencv_image(self, image_message, msg_type):
        image_np = None
        new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(
            self.intrinsic_coefficient_matrix,
            self.distortion_coefficient_matrix,
            (self.frame_width, self.frame_height),
            1,
            (self.frame_width, self.frame_height)
        )
        if msg_type=='compressed':
            np_arr = np.fromstring(image_message.data, np.uint8)
            image_np = cv.imdecode(np_arr, cv.IMREAD_COLOR)
        elif msg_type=='usb_raw':
            np_arr = np.frombuffer(image_message.data, dtype=np.uint8)
            image_np = np_arr.reshape((image_message.height, image_message.width, -1))
        grey_image = cv.cvtColor(src=image_np, code=cv.COLOR_BGR2GRAY)
        current_image = self.undistort_image(grey_image, new_camera_matrix)

        return current_image

    # helper function to make transformation matrix
    def make_transform_mat(self, translation, euler):
        rx, ry, rz = euler
        rotation = transf.euler_matrix(rx, ry, rz, axes='sxyz')
        translation = transf.translation_matrix(translation)
        return translation.dot(rotation)

    # method that parses camera intrinsics and computes 1) distortion coefficients and 2) intrinsic coefficients
    def parse_camera_intrinsics(self):
        calibration_file_path = self.calibration_file_path
        with open(calibration_file_path) as camera_calibration:
            data = yaml.load(camera_calibration, Loader=SafeLoader)

        if not self.controlled: # using the camera calibration file for the robot

            self.distortion_coefficient_matrix = np.array(data['distortion_coeffs'][0])
            self.intrinsic_coefficient_matrix = np.array(data['intrinsic_coeffs'][0]).reshape((3, 3))

        else: # using the camera calibration yaml for the lab iMAC
            camera_matrix_data = data['camera_matrix']['data']
            distortion_coeff_data = data['distortion_coefficients']['data']

            self.intrinsic_coefficient_matrix = np.array(camera_matrix_data).reshape((3, 3))
            self.distortion_coefficient_matrix = np.array(distortion_coeff_data).reshape((1, 5))

    """
    THIS SECTION CONTAINS ALL THE FUNCTIONS NEEDED FOR THE VISUAL ODOMETRY CALLBACK
    """

    # method to help visualize key points matching between current and previous key points
    def visualize_key_points_matching(self, previous_image, previous_descriptors, previous_keypoints,
                                      current_descriptors, current_key_points,
                                      current_image_with_keypoints_drawn):

        # Match descriptors.
        matches = self.bf.match(previous_descriptors, current_descriptors)
        # TODO: discuss the possible ways to sort the matches
        img3 = cv.drawMatches(previous_image, current_key_points, current_image_with_keypoints_drawn,
                              current_key_points, matches[:10], None,
                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        image_path = "/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/unit_testing_07242023/set5_ut_072423_matches.jpg"
        cv.imwrite(image_path, img3)

        cv.imshow('pattern', img3), cv.waitKey(5)

    # method to get the matches between 2 frames
    # Settings: using the BFMatcher (Brute Force); sorting depends on function call; default: sort = False
    def get_matches_between_two_frames(self, previous_key_points, previous_descriptors, current_key_points,
                                       current_descriptors):

        matches = None

        # initialize key points arrays
        top_previous_key_points = []
        top_current_key_points = []

        if self.mode == 'sift':
            matches = self.bf.match(previous_descriptors, current_descriptors)

        elif self.mode == 'knn_sift':
            matches = self.bf.knnMatch(previous_descriptors, current_descriptors, k=2)

        elif self.mode == 'flann':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary

            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(previous_descriptors, current_descriptors, k=2)

        elif self.mode == 'surf':
            matches = self.bf.knnMatch(previous_descriptors, current_descriptors, k=2)

        elif self.mode == 'orb':
            # use the brute force matcher to match features
            matches = self.bf.match(previous_descriptors, current_descriptors)
            # sort if needed
            matches = sorted(matches, key=lambda x: x.distance)

        passed_ratio_test = []
        # if we are not in orb mode, a ratio test is needed
        if self.mode != "orb":
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    passed_ratio_test.append([m])
        else:
            passed_ratio_test = matches

        # extract top 10 previous and current key points
        for i in range(len(passed_ratio_test)):
            train_index = passed_ratio_test[i][0].trainIdx  # index of the match in previous key points
            query_index = passed_ratio_test[i][0].queryIdx  # index of the match in current key points
            # print("train index {} and query index {} ".format(train_index, query_index))
            top_previous_key_points.append(previous_key_points[query_index])
            top_current_key_points.append(current_key_points[train_index])
        return matches, top_previous_key_points, top_current_key_points

    def get_transformation_between_two_frames(self, array_previous_key_points,
                                              array_current_key_points):

        # get the essential matrix
        self.essential_matrix, mask = cv.findEssentialMat(points1=array_previous_key_points,
                                                          points2=array_current_key_points,
                                                          cameraMatrix=self.intrinsic_coefficient_matrix,
                                                          method=cv.RANSAC, prob=0.999, threshold=1.0)
        # compute the relative position using the essential matrix, key points  using cv.relativepose
        points, relative_rotation, translation, mask = cv.recoverPose(E=self.essential_matrix,
                                                                      points1=array_previous_key_points,
                                                                      points2=array_current_key_points,
                                                                      cameraMatrix=self.intrinsic_coefficient_matrix)
        translation = translation.transpose()[0]
        # make the translation unit

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
    def previous_current_matching(self, top_previous_key_points, top_current_key_points,
                                  robot_previous_position_transformation):

        """"you can choose to visualize the points matched between every two frames by uncommenting this line"""""

        # convert previous_key_points and current_key_points into floating point arrays
        array_previous_key_points = cv.KeyPoint_convert(top_previous_key_points)

        # top 10 previous key points is a list of KeyPoint objects, array_previous_key points is a numpy array
        array_current_key_points = cv.KeyPoint_convert(top_current_key_points)

        # get the 4x4 homogenous transformation between previous image and the current image using the array list forms of previous and current key points
        prev_to_curr_transformation = self.get_transformation_between_two_frames(array_previous_key_points,
                                                                                 array_current_key_points)

        # calculate the current position using the previous-to-current homogenous transformation
        robot_current_position_transformation = robot_previous_position_transformation.dot(prev_to_curr_transformation)
        return robot_current_position_transformation, prev_to_curr_transformation # where is the robot now in VO frame

    def compute_current_image_elements(self, input_image):

        # find the keypoints and descriptors with ORB
        key_points, descriptors = self.feature_detector.detectAndCompute(input_image, None)
        # current key points output a list of KeyPoint elements, current descriptors is numpy array of (500 x 32 dimensions)
        image_with_keypoints_drawn = cv.drawKeypoints(input_image, key_points, None,
                                                      color=(0, 255, 0), flags=0)
        # current image with keypoints drawn is a numpy array

        return key_points, descriptors, image_with_keypoints_drawn

    """This is the main method for visual odometry calculations; calls other helper functions to compute immediate values"""

    # param: previous image, current image, and the 4x4 robot previous position (homogenous transformation matrix)
    def visual_odometry_calculations(self, previous_image, current_image, robot_previous_position_transformation):
        # get key points and descriptors for previous image
        previous_key_points, previous_descriptors, previous_image_with_keypoints_drawn = self.compute_current_image_elements(
            previous_image)

        # get current key points and descriptors for current image
        current_key_points, current_descriptors, current_image_with_keypoints_drawn = self.compute_current_image_elements(
            current_image)

        # print("length of prev kp {} and curr kp {}".format(len(previous_key_points), len(current_key_points)))
        # print("length of prev desc {} and curr desc {}".format(len(previous_descriptors), len(current_descriptors)))

        # from the above code we produce a length 500 current key points and a (500 height, 32 width) shaped current descriptors
        # get matches should return top 10 previous key points
        # TODO: check if we can determine number of matches based on distance to the marker (if distance > certain threshold, can get more)
        top_ten_matches, top_previous_key_points, top_current_key_points = self.get_matches_between_two_frames(
            previous_key_points=previous_key_points, previous_descriptors=previous_descriptors,
            current_key_points=current_key_points, current_descriptors=current_descriptors)

        robot_current_position_transformation, prev_to_curr_transformation = self.previous_current_matching(top_previous_key_points,
                                                                               top_current_key_points,
                                                                               robot_previous_position_transformation)

        return robot_current_position_transformation, prev_to_curr_transformation


# create the name function
if __name__ == '__main__':
    pass
