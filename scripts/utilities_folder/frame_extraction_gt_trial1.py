#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 7-24-2023
# Purpose: correct way to extract ground truth and convert into good frames

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


class FrameExtraction:
    def __init__(self, default_base_link_topic=DEFAULT_BASE_LINK_TOPIC, default_camera_topic=DEFAULT_CAMERA_TOPIC, starting_index=1):

        """
        by initializing vo, the image subscriber in hypothesis is activated
        uncomment the draw matches section that save the matched image
        """

        self.robot_frame_2_stamp = None
        self.robot_frame_1_stamp = None

        self.bridge = CvBridge()
        self.parse_camera_intrinsics()

        self.frame_one = None
        self.image_one = None
        self.frame_two = None
        self.image_two = None

        self.ground_truth_full_list_in_base_link = []
        self.ground_truth_list_cam_to_marker = []

        self.done_extracting = False

        self.ground_truth_frame_one = False
        self.ground_truth_frame_two = False
        self.ground_truth_index = 0
        self.detected_marker = False
        # self.vo = VisualOdometry()
        self.frame_dimensions = (1400, 1080)

        self.listener = tf.TransformListener()
        self.default_base_link_topic = default_base_link_topic
        self.default_camera_topic = default_camera_topic

        # finally, activate all the callbacks
        self.activate_callbacks()
        self.starting_index = starting_index


    def activate_callbacks(self):
        self.image_subscriber = rospy.Subscriber("/camera_array/cam0/image_raw/compressed", CompressedImage,
                                                 self.frame_extraction_callback, queue_size=1)
        self.ground_truth_subscriber = rospy.Subscriber("/bluerov_controller/ar_tag_detector", StagMarkers,
                                                        self.marker_callback,
                                                        queue_size=1)  # TODO: ask sam about the StagMarker posef

    def parse_camera_intrinsics(self):
        calibration_file_path = '/src/vis_odom/Parameters/camera_calibration.yaml'
        # rospy.loginfo("Parsing camera calibration from file {}".format(calibration_file_path))
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

    def frame_extraction_callback(self, robot_frame):
        # global current_image
        try:
            if not self.done_extracting:

                if self.frame_one is None:
                    self.frame_one = robot_frame
                    print("Here's the header for robot frame 1: {}".format(robot_frame.header.stamp))
                    self.robot_frame_1_stamp = robot_frame.header.stamp
                    self.image_one = self.rosframe_to_current_image(frame=robot_frame,
                                                                    frame_dimensions=self.frame_dimensions)
                    image_path = "/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/unit_testing_07262023/test_set_frame1.jpg"
                    cv.imwrite(image_path, self.image_one)
                    print("frame one extracted")

                else:
                    self.frame_two = robot_frame
                    print("Here's the header for robot frame 2: {}".format(robot_frame.header.stamp))
                    self.robot_frame_2_stamp = robot_frame.header.stamp
                    self.image_two = self.rosframe_to_current_image(frame=robot_frame,
                                                                    frame_dimensions=self.frame_dimensions)
                    image_path = "/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/unit_testing_07262023/test_set_frame2.jpg"

                    cv.imwrite(image_path, self.image_two)
                    print("frame two extracted")

                if self.frame_one is not None and self.frame_two is not None and self.starting_index==20:
                    self.done_extracting = True


        except CvBridgeError as e:
            print("finished extracting")
            print(e)

    """
    this method computes the base link to marker translation (ground truth) at a given frame
    """

    def compute_and_save_reading(self, markers):
        pose_position = markers[0].pose.pose.position  # access the pose position
        x, y, z = pose_position.x, pose_position.y, pose_position.z
        ground_truth_point = [x, y, z]  # create an (x,y,z) point object
        self.ground_truth_full_list_in_base_link.append(
            np.array(ground_truth_point))  # add the ground truth point into the list of ground truths

    """
    this method computes the camera to marker translation (ground truth) at a given frame
    """

    def compute_frame_camera_to_marker(self, markers):
        # TODO:
        # PART A:
        # for each frame, first get the pose translation and rotation info from base link
        # use tf transformations to get translation and rotation matrices
        # concatenate the two; the output is a homogenous transformation matrix 4x4, and yon now have base to marker

        bTm_translation = markers[0].pose.pose.position
        btm_orientation = markers[0].pose.pose.orientation
        print("we have btm translation {} and orientation {}".format(bTm_translation, btm_orientation))

        bTm_translation_array = np.array([bTm_translation.x, bTm_translation.y, bTm_translation.z])
        bTm_rotation_array = np.array([btm_orientation.x, btm_orientation.y, btm_orientation.z, btm_orientation.w])

        bTm_translation_mat = tf.transformations.translation_matrix(bTm_translation_array)
        bTm_orientation_mat = tf.transformations.quaternion_matrix(bTm_rotation_array)
        print("we have btm translation matrix {} and orientation matrix {}".format(bTm_translation, btm_orientation))

        btm_homogenous_translation_mat = tf.transformations.concatenate_matrices(bTm_translation_mat,
                                                                                 bTm_orientation_mat)
        print("we homogenous translation matrix {}".format(btm_homogenous_translation_mat))

        # PART B:
        # to get camera to baselink:
        # lookUpTransform produces a cam-2-baselink translation + rotation; repeat steps 2 - 3 in part A; produces 4x4 CTB
        # dot product: CTB dot BTM, you get CTM which is what the output of this function should be

        cTb_translation, cTb_rotation = self.listener.lookupTransform(self.default_base_link_topic,
                                                                      self.default_camera_topic,
                                                                      markers[0].header.stamp)
        print("we have cTb translation {} and orientation {}".format(cTb_translation, cTb_rotation))

        cTb_translation_mat = tf.transformations.translation_matrix(cTb_translation)
        cTb_orientation_mat = tf.transformations.quaternion_matrix(cTb_rotation)
        print("we have cTb translation matrix {} and orientation matrix {}".format(cTb_translation_mat,
                                                                                   cTb_orientation_mat))

        cTb_homogenous_translation_mat = tf.transformations.concatenate_matrices(cTb_translation_mat,
                                                                                 cTb_orientation_mat)
        print("we homogenous translation matrix {}".format(cTb_homogenous_translation_mat))

        # now get camera to marker transformation (4x4 homogenous transformation matrix)
        cam_to_marker_transformation = np.matmul(btm_homogenous_translation_mat, cTb_homogenous_translation_mat)

        print("Here is the camera to marker transformation: {}".format(cam_to_marker_transformation))
        self.ground_truth_list_cam_to_marker.append(cam_to_marker_transformation)

        return cam_to_marker_transformation

    """
    this method gets the marker to marker translation every two consecutive frames with marker readings
    """

    def get_translation_between_two_frames(self, frame1_cTm, frame2_cTm):
        inverse_frame2_cTm = np.linalg.inv(frame2_cTm)
        marker_transform = np.matmul(inverse_frame2_cTm, frame1_cTm)
        print("marker has translated {}".format(marker_transform))
        translation_only = np.array(
            [marker_transform.item(0, 3), marker_transform.item(1, 3), marker_transform.item(2, 3)])
        print("here is the translation decomposed: {}".format(translation_only))
        print("here is the lin alg norm of translation only {}".format(np.linalg.norm(translation_only)))
        unit_translation = translation_only / np.linalg.norm(translation_only)
        print("here is the unit vector translation {}".format(unit_translation))

        ## make translation a unit vector ***

    def marker_callback(self, msg):
        # callback function to access the ground truth data
        global frame1_cam2marker, frame2_cam2marker
        try:
            # print("the type of marker callback message received: {}".format(type(msg)))
            self.detected_marker = True
            # self.start_feature_matching = True
            markers = msg.markers  # get the marker information
            # print("Here's the header for ground truth: {}".format(markers[0].header.stamp))

            if len(markers) > 0 and self.detected_marker:
                for marker in markers:
                    if marker.id == 0:
                        if marker.header.stamp == self.robot_frame_1_stamp:
                            # print("currently processing the first frame")
                            frame1_cam2marker = self.compute_frame_camera_to_marker(markers)
                        # self.compute_and_save_reading(markers)

                        if marker.header.stamp == self.robot_frame_2_stamp:
                            frame2_cam2marker = self.compute_frame_camera_to_marker(markers)
                            # self.compute_and_save_reading(markers)

                        if len(self.ground_truth_list_cam_to_marker) == 2:
                            self.get_translation_between_two_frames(frame1_cam2marker, frame2_cam2marker)
                            self.ground_truth_subscriber.unregister()
                    else:
                        print("did not find marker 0")




            else:
                rospy.debug("No markers detected")
                self.detected_marker = False

        except Exception as e:
            self.detected_marker = False
            print("Could not process marker data!")
            print(e)
            pass

    def are_they_the_same(self):  # note: only two frames to compare
        if self.done_extracting:
            try:
                print("Are the two frames the same? {}".format(self.frame_one == self.frame_two))
                print("Are the two frames the same format? {}, {}".format(type(self.frame_one), type(self.frame_two)))

                cv.imshow("previous", self.frame_one)
                cv.imshow("current", self.frame_two)
            except:
                print("I cannot do it")

    # def getting_key_points(self):


def main(args):
    rospy.init_node('FrameExtractionNode', anonymous=True)
    while not rospy.is_shutdown():
        FrameExtraction()
    print("frame extraction activated")
    rospy.sleep(1)



# create the name function
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
