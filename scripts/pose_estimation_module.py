# Author: Ivy Aiwei Zhang
# Last updated: 08-06-2023
# Purpose: a python file containing matrix algebra helper functions

# ROS node messages
print("extracting 1) visual odometry and 2) ground truth from rosbag for trajectory evaluation")

import tf as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
# from numpy_quaternion import Quaternion


def transformation_from_translation_quaternion(translation, quaternion):
    # Convert the quaternion to a rotation matrix
    rotation_matrix = tf.quaternion_matrix(quaternion)[:3, :3]

    transformation_matrix = np.eye(4)  # Start with identity matrix
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation

    return transformation_matrix


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


def get_camera_to_camera_transformation(previous_cTm_transform, current_cTm_transform):
    inverse_current_cTm = np.linalg.inv(current_cTm_transform)
    camera_to_camera_transform = np.matmul(previous_cTm_transform, inverse_current_cTm)
    return camera_to_camera_transform


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


def get_velocity_between_timestamps(relative_position_change, previous_timestamp, current_timestamp):
    # get the incremental change in time
    time_change = current_timestamp-previous_timestamp

    # calculate the translation velocity
    translation = np.array([relative_position_change[0, 3], relative_position_change[1, 3], relative_position_change[2, 3]])
    translation_velocity = translation/time_change

    # calculate the rotational velocity
    rotation = relative_position_change[:3, :3]
    rotational_velocity = rotation/time_change

    # combine into a full velocity homogenous transformation matrix
    velocity_transformation = np.eye(4)
    velocity_transformation[:3, :3] = rotational_velocity
    velocity_transformation[:3, 3] = translation_velocity
    print("the relative position change is {} and with time change {}, the velocity is {}".format(relative_position_change, time_change, velocity_transformation))
    return velocity_transformation

def get_gt_vo_difference(gt_file_path, vo_file_path):
    ground_truth_data = np.genfromtxt(gt_file_path)
    vis_odom_data = np.genfromtxt(vo_file_path)
    for i in range(ground_truth_data.shape[0] - 1):
        ground_truth_quaternion = (
            ground_truth_data[i, 4], ground_truth_data[i, 5], ground_truth_data[i, 6], ground_truth_data[i, 7])
        gt_row, gt_pitch, gt_yaw = tf.transformations.euler_from_quaternion(ground_truth_quaternion)
        gt_euler = np.array([gt_row, gt_pitch, gt_yaw])

        vis_odom_quaternion = (vis_odom_data[i, 4], vis_odom_data[i, 5], vis_odom_data[i, 6], vis_odom_data[i, 7])
        vo_row, vo_pitch, vo_yaw = tf.transformations.euler_from_quaternion(vis_odom_quaternion)
        vo_euler = np.array([vo_row, vo_pitch, vo_yaw])

        gt_vo_difference = vo_euler - gt_euler
        return gt_vo_difference


def write_gt_vo_difference_to_file(gt_file_path, vo_file_path, output_file_path):
    ground_truth_data = np.genfromtxt(gt_file_path)
    vis_odom_data = np.genfromtxt(vo_file_path)
    with open(output_file_path, 'w') as file:
        for i in range(ground_truth_data.shape[0] - 1):
            timestamp = ground_truth_data[i, 0]
            ground_truth_quaternion = (
                ground_truth_data[i, 4], ground_truth_data[i, 5], ground_truth_data[i, 6], ground_truth_data[i, 7])
            gt_row, gt_pitch, gt_yaw = tf.transformations.euler_from_quaternion(ground_truth_quaternion)
            gt_euler = np.array([gt_row, gt_pitch, gt_yaw])

            vis_odom_quaternion = (vis_odom_data[i, 4], vis_odom_data[i, 5], vis_odom_data[i, 6], vis_odom_data[i, 7])
            vo_row, vo_pitch, vo_yaw = tf.transformations.euler_from_quaternion(vis_odom_quaternion)
            vo_euler = np.array([vo_row, vo_pitch, vo_yaw])

            gt_vo_difference = vo_euler - gt_euler
            print(gt_vo_difference)
            file.write("at timestamp {} the gt vo euler angle difference is {} \n".format(timestamp, gt_vo_difference))


def append_transformation_to_file(transformation_matrix, file_path):
    with open(file_path, 'a') as file:
        for row in transformation_matrix:
            row_str = ' '.join(str(value) for value in row)
            file.write(row_str + '\n')

def compute_gt_vo_translation_difference(gt_file_path, vo_file_path):
    ground_truth = np.genfromtxt(gt_file_path)
    vis_odom = np.genfromtxt(vo_file_path)

    ground_truth_translation = np.array([ground_truth[1], ground_truth[2], ground_truth[3]])
    vis_odom_translation = np.array([vis_odom[1], vis_odom[2], vis_odom[3]])
    translation_difference = vis_odom_translation - ground_truth_translation

    return [translation_difference[0], translation_difference[1], translation_difference[2]]



def visualize_gt_vo_translation_difference(translation_difference, plot_output_path):
    # Create a 3D plot

    plt.ion()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([translation_difference[0], translation_difference[0]], [translation_difference[1], translation_difference[1]], [translation_difference[2], translation_difference[2]], 'bo-')
    ax.scatter(translation_difference[0], translation_difference[1], translation_difference[2], c='r', marker='o', label='Vector 1')
    ax.scatter(translation_difference[0], translation_difference[1], translation_difference[2], c='g', marker='o', label='Vector 2')
    ax.plot([translation_difference[0], translation_difference[0]], [translation_difference[1], translation_difference[1]], [translation_difference[2], translation_difference[2]], 'b--')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Translation Difference Visualization')
    ax.legend()
    plt.savefig(plot_output_path, format='jpg', dpi=300)
    plt.show()

