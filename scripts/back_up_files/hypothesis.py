#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 7-05-2023
# this is the visual odometry node that processes information in the image_raw

#### ROS node messages
print("hello world")
import rospy
from sensor_msgs.msg import CompressedImage
from stag_ros.msg import StagMarkers
from geometry_msgs.msg import PoseStamped

#### other packages
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

VERBOSE = False

number_of_frames = 25075


### UTILITY FUNCTION FOR TRANSFORMTION MATRIX ###


class VisualOdometry:
    # global all_frames, previous_image, previous_key_points, previous_descriptors, current_frame, robot_position_list
    count = 0
    frame_height = 1080
    frame_width = 1400

    def __init__(self):
        self.current_image_with_keypoints_drawn = None
        self.current_descriptors = None
        self.current_key_points = None
        self.previous_matches_keypoints = None
        self.essential_matrix = None
        self.bridge = CvBridge()
        self.parse_camera_intrinsics()
        # do the matching here
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # subscribers
        self.image_subscriber = rospy.Subscriber("/camera_array/cam0/image_raw/compressed", CompressedImage,
                                                 self.image_callback, queue_size=1)
        self.ground_truth_subscriber = rospy.Subscriber("/bluerov_controller/ar_tag_detector", StagMarkers,
                                                        self.marker_callback, queue_size=1)
        # parameters needed for evaluating the image frames
        self.all_frames = []
        self.orb_feature_detector = cv.ORB_create()
        self.previous_image = None
        self.previous_key_points = None
        self.previous_descriptors = None
        self.current_frame = None
        self.current_frame = None
        # list to store the relevant positions
        self.robot_position_list = []  # calculated robot position list from opencv
        self.ground_truth_list = []  # list of ground truths from the ros node
        self.frame_translations = []  # maintains the translation between every consecutive frame
        self.marker_differences = []  # maintains the difference between the marker positions between every consecutive frames

        self.prev_transformation_matrix = self.make_transform_mat(translation=[0, 0, 0], euler=[0, 0, 0])
        # print("Previous transform matrix at initialization: {}".format(self.prev_transformation_matrix))
        self.robot_curr_position = self.make_transform_mat(translation=[0, 0, 0], euler=[0, 0, 0])
        # print("robot_curr_position at initialization: {}".format(self.robot_curr_position))

        self.detected_marker = False
        self.start_feature_matching = False

    def undistort_image(self, distorted_image, new_camera_matrix):
        current_image = cv.undistort(distorted_image, self.int_coeff_mtx, self.dist_coef_arr, None, new_camera_matrix)
        return current_image

    def rosframe_to_current_image(self, frame, frame_dimensions):
        # compute the camera matrix
        new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(
            self.int_coeff_mtx,
            self.dist_coef_arr,
            frame_dimensions,
            1,
            frame_dimensions
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
        rospy.loginfo("Parsing camera calibration from file {}".format(calibration_file_path))
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

        # rospy.loginfo(
        #     "Parsed camera calibration. Distortion: {}. Intrinsics: {}".format(self.dist_coef_arr, self.int_coeff_mtx))

    def marker_callback(self, msg):
        # callback function to access the ground truth data
        try:
            # print("the type of marker callback message received: {}".format(type(msg)))
            self.detected_marker = True
            self.start_feature_matching = True
            markers = msg.markers  # get the marker information

            if len(markers) > 0 and self.detected_marker:
                pose_position = markers[0].pose.pose.position  # access the pose position
                x, y, z = pose_position.x, pose_position.y, pose_position.z
                ground_truth_point = [x, y, z]  # create an (x,y,z) point object
                self.ground_truth_list.append(
                    np.array(ground_truth_point))  # add the ground truth point into the list of ground truths

                # compute the difference between two consecutive measurement of marker positions
                if len(self.ground_truth_list) > 1:
                    marker_translation = np.subtract(np.array(self.ground_truth_list[-1]),
                                                     np.array(self.ground_truth_list[-2]))
                    self.marker_differences.append(marker_translation)  # add the difference to the corresponding list
            else:
                rospy.debug("No markers detected")
                self.detected_marker = False
                self.start_feature_matching = False
        except Exception as e:
            self.detected_marker = False
            self.start_feature_matching = False
            print("Could not process marker data!")
            print(e)
            pass

    def visualize_key_points_matching(self, current_descriptors, current_key_points,
                                      current_image_with_keypoints_drawn):

        # Match descriptors.
        matches = self.bf.match(self.previous_descriptors, current_descriptors)
        matches = sorted(matches, key=lambda x:x.distance)

        img3 = cv.drawMatches(self.previous_image, current_key_points, current_image_with_keypoints_drawn,
                              current_key_points, matches[:10], None,
                              flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        image_path = "/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/frame1frame2_matches.jpg"
        cv.imwrite(image_path, img3)

        cv.imshow('pattern', img3), cv.waitKey(5)

    def get_matches_between_two_frames(self):
        self.previous_matches_keypoints = []
        self.current_matches_keypoints = []


        if self.previous_image is not None:
            matches = self.bf.match(self.previous_descriptors, self.current_descriptors)
            matches = sorted(matches, key=lambda x:x.distance)

            DMatches_Attributes = []
            for match in matches[:10]:
                match_dictionary = {}
                distance = match.distance
                image_index = match.imgIdx
                query_index = match.queryIdx
                train_index = match.trainIdx
                match_dictionary['imgIdx'] = image_index
                match_dictionary['distance'] = distance
                match_dictionary['queryIdx'] = query_index
                match_dictionary['trainIdx'] = train_index
                # print(self.previous_descriptors[train_index])
                
                self.current_matches_keypoints.append(self.current_key_points[train_index])
                self.previous_matches_keypoints.append(self.previous_key_points[query_index])

                # self.previous_image_descriptors.append(self.previous_descriptors)

                DMatches_Attributes.append(match_dictionary)

            print("Here are the top 10 matches: {}".format(DMatches_Attributes))
            # print("here are the current matches keypoints: {}".format(self.current_matches_keypoints))
            # print("here are the previous matches keypoints: {}".format(self.previous_matches_keypoints))
            return self.previous_matches_keypoints, self.current_matches_keypoints

        else:
            print("No available previous image")

    def get_translation_between_two_frames(self):

        # TODO matching first before findEssentialMat *****

        # get the essential matrix
        print("intrinsic coefficients", self.int_coeff_mtx)
        self.essential_matrix, mask = cv.findEssentialMat(points1=self.current_matches_keypoints,
                                                          points2=self.previous_matches_keypoints,
                                                          cameraMatrix=self.int_coeff_mtx,
                                                          method=cv.RANSAC, prob=0.999, threshold=1.0)

        # compute the relative position using the essential matrix, key points  using cv.relativepose
        points, relative_rotation, translation, mask = cv.recoverPose(E=self.essential_matrix,
                                                                      points1=self.current_matches_keypoints,
                                                                      points2=self.previous_matches_keypoints)
        translation = translation.transpose()[0]
        relative_rotation = np.array(relative_rotation)
        # decompose rotation matrix + find euler
        t = np.array([0, 0, 0])
        new_rotation_mat = np.vstack((np.hstack((relative_rotation, t[:, None])), [0, 0, 0, 1]))

        # get euler angles
        euler = transf.euler_from_matrix(new_rotation_mat, 'rxyz')

        # compute the current transformation matrix
        euler = np.array(euler)

        prev2curr_translation = self.make_transform_mat(translation=translation, euler=euler)
        # store previous to current translation in the corresponding list
        self.frame_translations.append(prev2curr_translation)
        print("translation between the two frames is : {}".format(prev2curr_translation))
        return prev2curr_translation



    def compute_current_image_elements(self, current_image):

        self.current_key_points = self.orb_feature_detector.detect(current_image, None)
        # print("current key points: ", type(current_key_points))

        # find the keypoints and descriptors with ORB
        self.current_key_points, self.current_descriptors = self.orb_feature_detector.compute(current_image,
                                                                                    self.current_key_points)
        self.current_image_with_keypoints_drawn = cv.drawKeypoints(current_image, self.current_key_points, None,
                                                              color=(0, 255, 0), flags=0)

        return self.current_key_points, self.current_descriptors, self.current_image_with_keypoints_drawn

    def previous_current_matching(self, previous_image):
        if previous_image is not None:
            # """"you can choose to visualize the points matched between every two frames by uncommenting this line"""""
            # self.visualize_key_points_matching(current_descriptors, current_key_points, current_image_with_keypoints_drawn)

            # convert previous_key_points and current_key_points into floating point arrays
            array_previous_key_points = cv.KeyPoint_convert(self.previous_matches_keypoints)
            array_current_key_points = cv.KeyPoint_convert(self.current_matches_keypoints)
            print("here are the current matches keypoints: {}".format(array_previous_key_points))
            print("here are the previous matches keypoints: {}".format(array_current_key_points))
            # print("the length of previous keypoints = {}, length of current = {}".format(len(self.previous_key_points), len(current_key_points)))

            # get the translation between previous image and the current image using the array list forms of previous and current key points
            prev2curr_translation = self.get_translation_between_two_frames()

            # calculate the current position using the previous-to-current translation
            self.robot_curr_position = self.robot_curr_position.dot(prev2curr_translation)
            _, _, _, robot_current_translation, _ = transf.decompose_matrix(self.robot_curr_position)
            self.robot_position_list.append(robot_current_translation)  # append to the corresponding list

    def image_callback(self, frame):
        if self.start_feature_matching:
            try:
                frame_dimensions = (VisualOdometry.frame_width, VisualOdometry.frame_height)

                # get the undistorted opencv current image
                current_image = self.rosframe_to_current_image(frame, frame_dimensions)
                # print("the format of previous image: {} \n and current image: {}".format(type(self.previous_image), type(current_image)))

                # get the relevant information needed for computing the translation
                current_key_points, current_descriptors, current_image_with_keypoints_drawn = self.compute_current_image_elements(
                    current_image)

                print("here are the previous key points: {}".format(type(self.previous_key_points)))

                 # starting from frame number two, calculate the matches:\
                self.get_matches_between_two_frames() # returns the top 10 matches between previous and current frames


                # starting from frame number two, calculate translation and current position
                self.previous_current_matching(self.previous_image)


                # update
                self.previous_image = current_image_with_keypoints_drawn
                self.previous_key_points = current_key_points  # same key points of PREVIOUS frame
                self.previous_descriptors = current_descriptors

            except CvBridgeError as e:
                print("CV bridge error")
                print(e)
        else:
            print("Pausing features matching")

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


def main(args):
    rospy.init_node('VisualOdometryNode', anonymous=True)
    vis_odom = VisualOdometry()
    print("visual odometry activated")
    rospy.sleep(1)

    try:
        vis_odom.setup_plots()
        while not rospy.is_shutdown():
            vis_odom.plot_position_information()

    except rospy.ROSInterruptException:
        rospy.logerr("Ros node interrupted")
        cv.destroyAllWindows()


# create the name function
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
