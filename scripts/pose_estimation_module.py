# Author: Ivy Aiwei Zhang
# Last updated: 08-06-2023
# Purpose: a python file containing matrix algebra helper functions

# ROS node messages
print("extracting 1) visual odometry and 2) ground truth from rosbag for trajectory evaluation")

import transformations as tf
import numpy as np


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
