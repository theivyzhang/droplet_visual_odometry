import numpy as np
import matplotlib.pyplot as plt
from math import atan2, asin
from mpl_toolkits.mplot3d import Axes3D  # Import the Axes3D module

import tf as tf

"""Step 1: extract translation and quaternion data from input txt files"""


class ExtractData:
    def __init__(self, traj_txt_file_path=' '):
        self.traj_txt_file_path = traj_txt_file_path
        self.translations = []
        self.quaternions = []
        self.timestamps = []
        self.output_translation_quaternion()

    def output_translation_quaternion(self):
        # Read data from the text file
        with open(self.traj_txt_file_path, 'r') as file:
            for line in file:
                parts = line.split()
                timestamp = float(parts[0])
                translation = [float(parts[i]) for i in range(1, 4)]
                quaternion = [float(parts[i]) for i in range(4, 8)]

                self.translations.append(translation)
                self.quaternions.append(quaternion)
                self.timestamps.append(timestamp)


"""Step 2: convert the information into 1) average 2) standard deviation separately"""


def get_stddevs(translations, quaternions):
    # Convert lists to numpy arrays for calculations
    translations = np.array(translations)
    quaternions = np.array(quaternions)

    # Calculate average and standard deviation for the translations
    std_translation = np.std(translations, axis=0)

    # Convert quaternions to Euler angles
    euler_angles = np.array([tf.transformations.euler_from_quaternion(q) for q in quaternions])

    # Calculate average and standard deviation of Euler angles
    std_euler_angles = np.std(euler_angles, axis=0)

    return std_translation, std_euler_angles


def get_averages(translations, quaternions):
    # Convert lists to numpy arrays for calculations
    translations = np.array(translations)
    quaternions = np.array(quaternions)

    # Calculate average for the translations
    avg_translation = np.mean(translations, axis=0)

    # Convert quaternions to Euler angles
    euler_angles = np.array([tf.transformations.euler_from_quaternion(q) for q in quaternions])

    # Calculate average and standard deviation of Euler angles
    avg_euler_angles = np.mean(euler_angles, axis=0)
    return avg_translation, avg_euler_angles, euler_angles


# Modify the Plot class to create 3D dot plots
def plot_and_save(list, category, plot_destination_path):
    plots = '/' + category + '_3d_dot_plot.jpg'

    fig = plt.figure(figsize=(7, 5))

    list_array = np.array(list)

    # Create 3D dot plot for translations
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # Use Axes3D for 3D plots
    ax1.scatter(list_array[:, 0], list_array[:, 1], list_array[:, 2], c='m', marker='o')
    ax1.plot(list_array[:, 0], list_array[:, 1], list_array[:, 2], c='r', label='Translation_Traj')
    ax1.scatter(list_array[0, 0], list_array[0, 1], list_array[0, 2], c='k', marker='o', label='First Average')
    ax1.text(list_array[0, 0], list_array[0, 1], list_array[0, 2], 'First', fontsize=10, verticalalignment='bottom')
    ax1.set_xlabel('tx')
    ax1.set_ylabel('ty')
    ax1.set_zlabel('tz')
    ax1.set_title(category + " translation")

    # Save the 3D dot plots as image files
    plt.savefig(plot_destination_path + plots)
    plt.show()


print("starting data analysis")

"""
SECTION FOR VARIABLE INPUT
"""
destination_directory = '/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/unit_testing_controlled/controlled_usb_rosbot/backward-1'
ground_truth_positions_path = destination_directory + '/stamped_ground_truth_relative.txt'
vis_odom_positions_path = destination_directory + '/stamped_traj_estimate.txt'
ground_truth_absolute_path = destination_directory + '/stamped_ground_truth_absolute.txt'
ground_truth_inverse_path = destination_directory + '/stamped_ground_truth_inversed.txt'
# TODO:get aboslute position translation + quaternion
"""
SECTION FOR GROUND TRUTH
"""
# get the ground truth info + plot
gt_position_data = ExtractData(ground_truth_positions_path)
gt_position_translations = gt_position_data.translations
gt_position_quaternions = gt_position_data.quaternions
# compute the averages for ground truth
gt_avg_translation, gt_avg_euler_angles, gt_euler_angles = get_averages(translations=gt_position_translations,
                                                                        quaternions=gt_position_quaternions)
# compute the standard deviations for ground truth
gt_std_translation, gt_std_euler_angles = get_stddevs(translations=gt_position_translations,
                                                      quaternions=gt_position_quaternions)

"""
SECTION FOR VISUAL ODOMETRY
"""
# get the ground truth info + plot
vo_data = ExtractData(vis_odom_positions_path)
vo_position_translations = vo_data.translations
vo_position_quaternions = vo_data.quaternions
# compute the averages for ground truth
vo_avg_translation, vo_avg_euler_angles, vo_euler_angles = get_averages(translations=vo_position_translations,
                                                                        quaternions=vo_position_quaternions)
# compute the standard deviations for ground truth
vo_std_translation, vo_std_euler_angles = get_stddevs(translations=vo_position_translations,
                                                      quaternions=vo_position_quaternions)

"""
SECTION FOR GROUND TRUTH ABSOLUTE
"""
gt_abs_data = ExtractData(ground_truth_absolute_path)
gt_abs_translations = gt_abs_data.translations
gt_abs_quaternions = gt_abs_data.quaternions

"""
SECTION FOR GROUND TRUTH INVERSE
"""
gt_inv_data = ExtractData(ground_truth_inverse_path)
gt_inv_translations = gt_inv_data.translations
gt_inv_quaternions = gt_inv_data.quaternions


"""Plotting"""
plot_and_save(list=gt_inv_translations, category="ground_truth_inverse",
              plot_destination_path=destination_directory)
# plot the ground truth and visual odometry plots
plot_and_save(list=gt_position_translations, category="ground_truth_relative",
              plot_destination_path=destination_directory)
# plot_and_save(list=vo_position_translations, category="vo_estimates",
#               plot_destination_path=destination_directory)
plot_and_save(list = gt_abs_translations, category="ground_truth_absolute", plot_destination_path=destination_directory)
