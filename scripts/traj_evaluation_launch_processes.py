#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 08-04-2023
# Purpose: a process python file that concurrently runs visual odometry and ground truth extraction

# ROS node messages
print("extracting 1) visual odometry and 2) ground truth from rosbag for trajectory evaluation")

import sys
import os
import csv
import rosbag
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

from visual_odometry_v3 import VisualOdometry
from trajectory_evaluation_vis_odom_extraction import TrajectoryEvaluation_WriteData
from trajectory_evaluation_ground_truth import TrajEvalGroundTruth

import tf as tf


class Activate_GT_VO_Processes:
    def __init__(self, bag_file_path=None, gt_output_file_path=''):
        self.previous_image = None
        self.current_image = None
        self.first_topic_found = ""
        self.valid_count = 0

        self.vo_tf_list = []
        self.gt_tf_list = []

        # self.visual_odom_functions =
        self.bag_file_path = bag_file_path
        self.gt_output_file_path = gt_output_file_path
        self.visual_odometry = VisualOdometry()  # default settings of visual odometry; visual odometry functions
        # self.ground_truth_extraction = TrajEvalGroundTruth(loop=True, total_number_of_frames=200,
        #                                                                output_file_path=self.gt_output_file_path,
        #                                                                start_subscriber=True,
        #                                                                activate_image_callback=True,
        #                                                                activate_marker_reading=True
        #                                                                )  # ground truth extraction processes
        self.extract_vo_gt_data()

    def extract_vo_gt_data(self):
        with rosbag.Bag(self.bag_file_path, "r") as bag:
            tmp_gt_tf = None
            for topic, bag_message, timestamp in bag.read_messages(topics=['/camera_array/cam0/image_raw/compressed',
                                                                           '/bluerov_controller/ar_tag_detector'],
                                                                 ):
                print("{} {}".format(topic, timestamp.to_sec()))



                if self.first_topic_found:
                    if topic != self.first_topic_found:
                        # valid

                        # Saving ground truth
                        if tmp_gt_tf is None:
                            tmp_gt_tf = bag_message
                        # TODO have the message already in the form of a transformation
                        self.gt_tf_list.append(tmp_gt_tf)
                        tmp_gt_tf = None # Reset.

                        self.valid_count += 1
                        if (self.valid_count > 1):
                            if self.current_image is None:
                                self.current_image = bag_message
                            # VO
                            current_image = self.visual_odometry.rosframe_to_current_image(self.current_image)
                            previous_image = self.visual_odometry.rosframe_to_current_image(self.previous_image)

                            vo_tf = self.visual_odometry.visual_odometry_calculations(current_image, previous_image)
                            # TODO: another module that does the calculations to have the same global reference frame ***

                            self.vo_tf_list.append(vo_tf)

                            self.previous_image = self.current_image
                            self.current_image = None
                        else:
                            if "image" in topic:
                                self.previous_image = bag_message
                                vo_tf = np.eye((4,4)) # TODO ensure consistency with the other vo_tf
                                self.vo_tf_list.append(vo_tf)

                        self.first_topic_found = ""
                    else:
                        self.first_topic_found = topic
                        if "image" in topic:
                            if self.valid_count < 1:
                                self.previous_image = bag_message
                            elif self.valid_count == 1:
                                self.current_image = bag_message
                        else:
                            tmp_gt_tf = bag_message

                else:
                    self.first_topic_found = topic
                    if "image" in topic:
                        if self.valid_count < 1:
                            self.previous_image = bag_message
                        elif self.valid_count == 1:
                            self.current_image == bag_message
                        else:
                            self.current_image = bag_message







                # Assuming that the marker pose timestamp is greater than the timestamp of the corresponding image.


                # if self.starting_time is None:
                #     self.starting_time = bag_message.header.stamp
                #
                # print("got here", topic)
                #
                # if topic == '/camera_array/cam0/image_raw/compressed':
                #     print("receiving an instance of compressed images message")
                #     if self.previous_image is not None:
                #         print("we have robot current position at: ", self.visual_odometry.robot_curr_position)
                #         # TrajectoryEvaluation_WriteData()
                #     self.previous_image = bag_message
                # elif topic == '/bluerov_controller/ar_tag_detector':
                #     print("receiving an instance of stag markers bag message")
                #     print(self.ground_truth_extraction.ground_truth_list_cam_to_marker)
                #
                # if self.starting_time is not None:
                #     elapsed_time = (bag_message.header.stamp - self.starting_time).to_sec()
                #     print(elapsed_time)


def main(args):
    # rospy.init_node('TrajEvalLaunchProcess', anonymous=True)
    bag_file_path = '/home/ivyz/Documents/8-31-system-trials_2021-07-28-16-33-22.bag'
    gt_output_file_path = '/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/stamped_ground_truth_2.txt'
    Activate_GT_VO_Processes(bag_file_path=bag_file_path, gt_output_file_path=gt_output_file_path)

    # while not rospy.is_shutdown():
    #     FrameExtraction()
    print("trajectory evaluation processes activated")
    rospy.sleep(1)


# create the name function
if __name__ == '__main__':
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
