import numpy as np
import matplotlib.pyplot as plt
from math import atan2, asin
from mpl_toolkits.mplot3d import Axes3D  # Import the Axes3D module

import tf as tf
import sys

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

    bbox = [
        [
            min(x[0] for x in list),
            min(x[1] for x in list),
            min(x[2] for x in list),
        ],
        [
            max(x[0] for x in list),
            max(x[1] for x in list),
            max(x[2] for x in list),
        ]
    ]

    print("Bounding box extent is {}".format(np.array(bbox[1]) - np.array(bbox[0])))

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
    ax1.set_aspect('equal')

    # Save the 3D dot plots as image files
    plt.savefig(plot_destination_path + plots)
    plt.show()


def main(experiment_sample):


    print("starting data analysis")

    """
    SECTION FOR VARIABLE INPUT
    """
    destination_directory = '/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/unit_testing_controlled/controlled_usb_rosbot/'+experiment_sample
    ground_truth_absolute_path = destination_directory + '/stamped_ground_truth_absolute.txt'
    ground_truth_relative_path = destination_directory + '/stamped_ground_truth_relative.txt'
    vis_odom_absolute_path = destination_directory + '/stamped_traj_estimate_absolute.txt'
    vis_odom_relative_path = destination_directory + '/stamped_traj_estimate_relative.txt'

    # TODO:get aboslute position translation + quaternion
    """
    SECTION FOR GROUND TRUTH ABSOLUTE + RELATIVE
    """
    # get the ground truth info + plot
    gt_position_data = ExtractData(ground_truth_absolute_path)
    gt_position_translations = gt_position_data.translations

    gt_relative_data = ExtractData(ground_truth_relative_path)
    gt_relative_translations = gt_relative_data.translations

    """
    SECTION FOR VISUAL ODOMETRY ABSOLUTE + RELATIVE
    """
    # get the vis odom info + plot
    vo_data = ExtractData(vis_odom_absolute_path)
    vo_position_translations = vo_data.translations

    vo_relative_data = ExtractData(vis_odom_relative_path)
    vo_relative_translations = vo_relative_data.translations


    """Plotting"""
    plot_and_save(list = gt_position_translations, category="ground_truth_absolute", plot_destination_path=destination_directory)
    plot_and_save(list = gt_relative_translations, category="ground_truth_relative", plot_destination_path=destination_directory)
    plot_and_save(list = vo_position_translations, category="vis_odom_absolute", plot_destination_path=destination_directory)
    plot_and_save(list = vo_relative_translations, category="vis_odom_relative", plot_destination_path=destination_directory)


# create the name function
if __name__ == '__main__':
    try:
        experiment_sample = sys.argv[1]
        main(experiment_sample=experiment_sample)
    except:
        pass
