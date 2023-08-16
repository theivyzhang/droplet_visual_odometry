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

VERBOSE = False

number_of_frames = 25075

DEFAULT_STARTING_ROBOT_TRANSLATION = [0, 0, 0]
DEFAULT_STARTING_ROBOT_EULER = [0, 0, 0]


class VisualOdometry:
    # global all_frames, previous_image, previous_key_points, previous_descriptors, current_frame, robot_position_list

    def __init__(self, starting_translation=None, starting_euler=None, to_sort=False, mode="ORB"):
        if starting_euler is None:
            starting_euler = DEFAULT_STARTING_ROBOT_EULER
        if starting_translation is None:
            starting_translation = DEFAULT_STARTING_ROBOT_TRANSLATION

        # Assumes that message is compressed, thus requiring frame width and height
        self.frame_height = 1080
        self.frame_width = 1400
        # the translation and euler at the start of the program
        self.starting_translation = starting_translation
        self.starting_euler = starting_euler

        # global variables
        self.robot_current_translation = None
        self.essential_matrix = None

        # parse the camera intrinsics before starting anything else
        self.parse_camera_intrinsics()

        # model parameters
        self.to_sort = to_sort  # whether we want to sort the frames or not
        self.mode = mode

        # get the parameters needed under the specified mode
        self.feature_detector, self.norm_type, self.cross_check = self.return_feature_matching_parameters(mode)

        # parameters needed for evaluating the image frames

        self.bf = cv.BFMatcher(normType=self.norm_type, crossCheck=self.cross_check)

        # list to store the relevant positions
        self.robot_position_list = []  # calculated robot position list from opencv
        self.ground_truth_list = []  # list of ground truths from the ros node
        self.frame_translations = []  # maintains the translation between every consecutive frame
        self.marker_differences = []  # maintains the difference between the marker positions between every consecutive frames
        self.matches_dictionary = []  # list of dictionaries

        # initialize current position of the robot based on the translation and euler at the start of the program
        self.robot_curr_position = self.make_transform_mat(translation=self.starting_translation,
                                                           euler=self.starting_euler)

        self.detected_marker = False

    """
    THIS IS THE SECTION CONTAINING ALL THE UTILITY FUNCTIONS
    """

    def return_feature_matching_parameters(self, mode):
        global feature_detector, norm_type, cross_check
        if mode.lower() == "orb":
            feature_detector = cv.ORB_create()
            norm_type = cv.NORM_HAMMING
            cross_check = True
        elif mode.lower() == "sift" or mode.lower() == 'flann' or mode.lower()=='knn_sift':
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
        current_image = cv.undistort(distorted_image, self.int_coeff_mtx, self.dist_coef_arr, None, new_camera_matrix)
        return current_image

    # method to convert a ros frame to a compatible opencv image format
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

    # helper function to make transformation matrix
    def make_transform_mat(self, translation, euler):
        rx, ry, rz = euler
        rotation = transf.euler_matrix(rx, ry, rz, axes='sxyz')
        translation = transf.translation_matrix(translation)
        return translation.dot(rotation)

    # method that parses camera intrinsics and computes 1) distortion coefficients and 2) intrinsic coefficients
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

        if self.mode == 'sift':
            return self.get_sift_matches(previous_key_points, previous_descriptors, current_key_points,
                                         current_descriptors)
        elif self.mode =='knn_sift':
            return self.get_knn_sift_matches(previous_key_points, previous_descriptors, current_key_points,
                                         current_descriptors)
        elif self.mode == 'flann':
            return self.get_sift_flann_matches(previous_key_points, previous_descriptors, current_key_points, current_descriptors)
        elif self.mode == 'surf':
            return self.get_surf_matches(previous_key_points, previous_descriptors, current_key_points, current_descriptors)
        elif self.mode == 'orb':
            return self.get_orb_matches(previous_key_points, previous_descriptors, current_key_points,
                                        current_descriptors)


    # TODO: bf matching with orb, returning only top 100 out of 500
    def get_orb_matches(self, previous_key_points, previous_descriptors, current_key_points,
                        current_descriptors):
        # initialize key points arrays
        top_ten_previous_key_points = []
        top_ten_current_key_points = []

        # use the brute force matcher to match features
        matches = self.bf.match(previous_descriptors, current_descriptors)
        # sort if needed
        matches = sorted(matches, key=lambda x: x.distance)

        # extract top 10 previous and current key points
        for i in range(len(matches[:100])):
            train_index = matches[i].trainIdx  # index of the match in previous key points
            query_index = matches[i].queryIdx  # index of the match in current key points
            print("train index {} and query index {} ".format(train_index, query_index))
            top_ten_previous_key_points.append(previous_key_points[train_index])
            top_ten_current_key_points.append(current_key_points[query_index])
        return matches[:10], top_ten_previous_key_points, top_ten_current_key_points  # TODO parameter for top-k matches

    def get_sift_matches(self, previous_key_points, previous_descriptors, current_key_points,
                         current_descriptors):

        top_ten_previous_key_points = []
        top_ten_current_key_points = []

        matches = self.bf.match(previous_descriptors, current_descriptors)

        # print("here are the knn matches object types: {}".format(type(matches[0])))

        # TODO: CHECK - for SIFT, currently using ALL good matches with no filter
        # extract top 10 previous and current key points
        for i in range(len(matches[:100])):
            train_index = matches[i].trainIdx  # index of the match in previous key points
            query_index = matches[i].queryIdx  # index of the match in current key points
            # print("train index {} and query index {} ".format(train_index, query_index))
            top_ten_previous_key_points.append(previous_key_points[query_index])
            top_ten_current_key_points.append(current_key_points[train_index])

        # print("the ones that passed ratio test: {}".format(passed_ratio_test))

        return matches[:10], top_ten_previous_key_points, top_ten_current_key_points  # TODO parameter for top-k matches

    # TODO: bf matching with sift descriptors and ratio test
    # RESULTS under KNN_SIFT_08152023
    def get_knn_sift_matches(self, previous_key_points, previous_descriptors, current_key_points,
                         current_descriptors):

        top_ten_previous_key_points = []
        top_ten_current_key_points = []

        matches = self.bf.knnMatch(previous_descriptors, current_descriptors, k=2)

        # print("here are the knn matches object types: {}".format(type(matches[0])))

        passed_ratio_test = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                passed_ratio_test.append([m])

        # TODO: CHECK - for SIFT, currently using ALL good matches with no filter
        # extract top 10 previous and current key points
        for i in range(len(passed_ratio_test)):
            train_index = passed_ratio_test[i][0].trainIdx  # index of the match in previous key points
            query_index = passed_ratio_test[i][0].queryIdx  # index of the match in current key points
            # print("train index {} and query index {} ".format(train_index, query_index))
            top_ten_previous_key_points.append(previous_key_points[query_index])
            top_ten_current_key_points.append(current_key_points[train_index])

        # print("the ones that passed ratio test: {}".format(passed_ratio_test))

        return matches[:10], top_ten_previous_key_points, top_ten_current_key_points  # TODO parameter for top-k matches

    # TODO: check sift FLANN method

    def get_sift_flann_matches(self, previous_key_points, previous_descriptors, current_key_points, current_descriptors):
        top_ten_previous_key_points = []
        top_ten_current_key_points = []

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) # or pass empty dictionary

        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(previous_descriptors,current_descriptors,k=2)

        passed_ratio_test = []

        for m, n in matches:
            # print("m {}, n {}".format(m, n))
            if m.distance < 0.7 * n.distance:
                passed_ratio_test.append([m])

        # TODO: CHECK - for FLANN, currently using ALL good matches with no filter
        # extract top 10 previous and current key points
        for i in range(len(passed_ratio_test)):
            train_index = passed_ratio_test[i][0].trainIdx  # index of the match in previous key points
            query_index = passed_ratio_test[i][0].queryIdx  # index of the match in current key points
            # print("train index {} and query index {} ".format(train_index, query_index))
            # print("train {} and query {}".format(train_index, query_index))
            top_ten_previous_key_points.append(previous_key_points[query_index])
            top_ten_current_key_points.append(current_key_points[train_index])

        # print("the ones that passed ratio test: {}".format(passed_ratio_test))

        return matches[:10], top_ten_previous_key_points, top_ten_current_key_points  # TODO parameter for top-k matches

    # TODO: bf matching with sift descriptors and ratio test
    # RESULTS under KNN_SIFT_08152023
    def get_surf_matches(self, previous_key_points, previous_descriptors, current_key_points,
                         current_descriptors):

        top_ten_previous_key_points = []
        top_ten_current_key_points = []

        matches = self.bf.knnMatch(previous_descriptors, current_descriptors, k=2)

        # print("here are the knn matches object types: {}".format(type(matches[0])))

        passed_ratio_test = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                passed_ratio_test.append([m])

        # TODO: CHECK - for SIFT, currently using ALL good matches with no filter
        # extract top 10 previous and current key points
        for i in range(len(passed_ratio_test)):
            train_index = passed_ratio_test[i][0].trainIdx  # index of the match in previous key points
            query_index = passed_ratio_test[i][0].queryIdx  # index of the match in current key points
            # print("train index {} and query index {} ".format(train_index, query_index))
            top_ten_previous_key_points.append(previous_key_points[query_index])
            top_ten_current_key_points.append(current_key_points[train_index])

        # print("the ones that passed ratio test: {}".format(passed_ratio_test))

        return matches[:10], top_ten_previous_key_points, top_ten_current_key_points  # TODO parameter for top-k matches


    # TODO: CHECKED - PREV CURR
    def get_transformation_between_two_frames(self, array_previous_key_points,
                                              array_current_key_points):

        # get the essential matrix
        self.essential_matrix, mask = cv.findEssentialMat(points1=array_current_key_points,
                                                          points2=array_previous_key_points,
                                                          cameraMatrix=self.int_coeff_mtx,
                                                          method=cv.RANSAC, prob=0.999, threshold=1.0)
        # TODO: discuss the need for setting maxIters -> maximum number of robust method iterations
        # compute the relative position using the essential matrix, key points  using cv.relativepose
        points, relative_rotation, translation, mask = cv.recoverPose(E=self.essential_matrix,
                                                                      points1=array_current_key_points,
                                                                      points2=array_previous_key_points,
                                                                      cameraMatrix=self.int_coeff_mtx)
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

        # get the 4x4 homogenous transformation between previous image and the current image using the array list forms of previous and current key points
        prev_to_curr_transformation = self.get_transformation_between_two_frames(array_previous_key_points,
                                                                                 array_current_key_points)

        # calculate the current position using the previous-to-current homogenous transformation
        robot_current_position_transformation = robot_previous_position_transformation.dot(prev_to_curr_transformation)
        return robot_current_position_transformation

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

        print("length of prev kp {} and curr kp {}".format(len(previous_key_points), len(current_key_points)))
        print("length of prev desc {} and curr desc {}".format(len(previous_descriptors), len(current_descriptors)))

        # from the above code we produce a length 500 current key points and a (500 height, 32 width) shaped current descriptors
        # get matches should return top 10 previous key points
        # TODO: check if we can determine number of matches based on distance to the marker (if distance > certain threshold, can get more)
        top_ten_matches, top_10_previous_key_points, top_10_current_key_points = self.get_matches_between_two_frames(
            previous_key_points=previous_key_points, previous_descriptors=previous_descriptors,
            current_key_points=current_key_points, current_descriptors=current_descriptors)

        robot_current_position_transformation = self.previous_current_matching(top_10_previous_key_points,
                                                                               top_10_current_key_points,
                                                                               robot_previous_position_transformation)

        return robot_current_position_transformation


# create the name function
if __name__ == '__main__':
    pass
