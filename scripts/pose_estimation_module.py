# Author: Ivy Aiwei Zhang
# Last updated: 08-06-2023
# Purpose: a python file containing matrix algebra helper functions

# ROS node messages
print("extracting 1) visual odometry and 2) ground truth from rosbag for trajectory evaluation")

import transformations as tf
import numpy as np
import tf as tf


def translation_from_transformation_matrix(transformation_matrix):
    translation = [transformation_matrix[0, 3], transformation_matrix[1, 3], transformation_matrix[2, 3]]
    return translation

def rotation_matrix_to_quaternion(rotation_matrix):
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

def quaternion_from_transformation_matrix(transformation_matrix):
    rotation_matrix = transformation_matrix[:3, :3]
    # print("here is the rotation matrix: {}".format(rotation_matrix))
    quaternion_rep = rotation_matrix_to_quaternion(rotation_matrix)
    # print("here is the quaternion representation: {}".format(quaternion_rep))
    return quaternion_rep

def get_marker_to_marker_transformation(previous_cTm_transform, current_cTm_transform):
    inverse_previous_cTm = np.linalg.inv(previous_cTm_transform)
    marker_to_marker_transform = np.matmul(inverse_previous_cTm, current_cTm_transform)
    return marker_to_marker_transform

def write_to_output_file(output_file_path, timestamp, translation, quaternion):
    # write the information needed for the trajectory evaluation
    with open(output_file_path, 'a') as file:
        file.write(str(timestamp) + " " + str(translation[0]) + " " + str(
            translation[1]) + " " + str(translation[2]) + " "
                   + str(quaternion[0]) + " " + str(quaternion[1]) + " " + str(
            quaternion[2]) + " " + str(quaternion[3]) + " " + "\n")

def clear_txt_file_contents(file_path):
    with open(file_path, "w") as file:
        file.truncate()


def get_gt_vo_difference(gt_file_path, vo_file_path):
    ground_truth_data = np.genfromtxt(gt_file_path)
    vis_odom_data = np.genfromtxt(vo_file_path)
    for i in range(ground_truth_data.shape[0]-1):

         ground_truth_quaternion = (ground_truth_data[i, 4], ground_truth_data[i, 5], ground_truth_data[i, 6], ground_truth_data[i, 7])
         gt_row, gt_pitch, gt_yaw = tf.transformations.euler_from_quaternion(ground_truth_quaternion)
         gt_euler = np.array([gt_row, gt_pitch, gt_yaw])

         vis_odom_quaternion = (vis_odom_data[i, 4], vis_odom_data[i, 5], vis_odom_data[i, 6], vis_odom_data[i, 7])
         vo_row, vo_pitch, vo_yaw = tf.transformations.euler_from_quaternion(vis_odom_quaternion)
         vo_euler = np.array([vo_row, vo_pitch, vo_yaw])

         gt_vo_difference =  vo_euler - gt_euler
         print("difference between gt vo at timestamp {} is {}".format(ground_truth_data[i, 0], gt_vo_difference))




