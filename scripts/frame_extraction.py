#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 7-3-2023
# Purpose: extracts 2 adjacent frames from a running rosbag, then gets the fiducial marker information

# ROS node messages
print("hello world")

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

from hypothesis2 import VisualOdometry as vo


class FrameExtraction:
    def __init__(self):

        """
        by initializing vo, the image subscriber in hypothesis is activated
        uncomment the draw matches section that save the matched image
        """

        self.image_subscriber = rospy.Subscriber("/camera_array/cam0/image_raw/compressed", CompressedImage,
                                                 self.frame_extraction_callback, queue_size=1)
        self.ground_truth_subscriber = rospy.Subscriber("/bluerov_controller/ar_tag_detector", StagMarkers,
                                                        self.marker_callback, queue_size=1) # TODO: ask sam about the StagMarker pose
        self.bridge = CvBridge()
        self.parse_camera_intrinsics()

        self.frame_one = None
        self.image_one = None
        self.frame_two = None
        self.image_two = None

        self.ground_truth_full_list = []

        self.done_extracting = False

        self.ground_truth_frame_one = False
        self.ground_truth_frame_two = False
        self.ground_truth_index = 0
        self.detected_marker = False
        # self.vo = VisualOdometry()
        self.frame_dimensions = (vo.frame_width, vo.frame_height)

    def parse_camera_intrinsics(self):
        calibration_file_path = '/home/ivyz/Documents/ivy_workspace/src/vis_odom/Parameters/camera_calibration.yaml'
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
                    self.image_one = self.rosframe_to_current_image(frame=robot_frame,
                                                                       frame_dimensions=self.frame_dimensions)
                    image_path = "/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/robot_frame_one.jpg"
                    cv.imwrite(image_path, self.image_one)
                    print("frame one extracted")

                else:
                    self.frame_two = robot_frame
                    self.image_two = self.rosframe_to_current_image(frame=robot_frame,
                                                                       frame_dimensions=self.frame_dimensions)
                    image_path = "/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/images/robot_frame_two.jpg"

                    cv.imwrite(image_path, self.image_two)
                    print("frame two extracted")

                if self.frame_one is not None and self.frame_two is not None:
                    self.done_extracting = True




        except CvBridgeError as e:
            print("finished extracting")
            print(e)

    def marker_callback(self, msg):
        # callback function to access the ground truth data
        try:
            # print("the type of marker callback message received: {}".format(type(msg)))
            self.detected_marker = True
            # self.start_feature_matching = True
            markers = msg.markers  # get the marker information

            if len(markers) > 0 and self.detected_marker:
                pose_position = markers[0].pose.pose.position  # access the pose position
                x, y, z = pose_position.x, pose_position.y, pose_position.z
                ground_truth_point = [x, y, z]  # create an (x,y,z) point object
                self.ground_truth_full_list.append(
                    np.array(ground_truth_point))  # add the ground truth point into the list of ground truths

                if len(self.ground_truth_full_list) == 2:
                    ground_truth_data = open("ground_truth_data.txt","w")
                    marker_translation = np.subtract(np.array(self.ground_truth_full_list[-1]),
                                                     np.array(self.ground_truth_full_list[-2]))
                    print("between the two frames, we have frame one ground truth {} and frame two ground truth{}"
                          .format(self.ground_truth_full_list[-2], self.ground_truth_full_list[-1]))
                    print("between the two adjacent frames, the ground truth translation is {}".format(
                        marker_translation))

                    ground_truth_data.write(str(marker_translation[0])+" "+str(marker_translation[1])+" "+str(marker_translation[2]))
                    ground_truth_data.close()


            else:
                rospy.debug("No markers detected")
                self.detected_marker = False

        except Exception as e:
            self.detected_marker = False
            print("Could not process marker data!")
            print(e)
            pass


    # def unit_testing_visual_odometry(self):
    #     try:
    #         if self.done_extracting:
    #             self.vo.previous_image =
    #     except:
    #         print("Unable to execute visual odometry")

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
    FrameExtraction()
    print("frame extraction activated")
    rospy.sleep(1)
    # extract_frames.print_ground_truth_information()
    # print(fe.vo.matches_dictionary)


# create the name function
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
