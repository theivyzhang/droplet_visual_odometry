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
class Plot:
    def __init__(self, translations, euler_angles, plot_destination_path=''):
        self.translations = translations
        self.euler_angles = euler_angles
        self.plot_destination_path = plot_destination_path

    def plot_and_save(self, category):
        translations_euler_plots = '/'+category+'_3d_dot_plot.jpg'

        fig = plt.figure(figsize=(10, 5))

        translation_array = np.array(self.translations)
        euler_array = np.array(self.euler_angles)

        # Create 3D dot plot for translations
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')  # Use Axes3D for 3D plots
        ax1.scatter(translation_array[:, 0],translation_array[:, 1], translation_array[:, 2], c='b', marker='o')
        ax1.set_xlabel('tx')
        ax1.set_ylabel('ty')
        ax1.set_zlabel('tz')
        ax1.set_title("translation")


        # Create 3D dot plot for Euler angles
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')  # Use Axes3D for 3D plots
        ax2.scatter(euler_array[:, 0], euler_array[:, 1], euler_array[:, 2], c='r', marker='x')
        ax2.set_xlabel('Roll')
        ax2.set_ylabel('Pitch')
        ax2.set_zlabel('Yaw')
        ax2.set_title("euler")

        plt.tight_layout()

        # Save the 3D dot plots as image files
        plt.savefig(self.plot_destination_path + translations_euler_plots)


print("starting data analysis")

"""
SECTION FOR VARIABLE INPUT
"""
destination_directory = '/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/unit_testing_controlled/controlled_usb_rosbot_zoomin_move_1'
ground_truth_file_path = '/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/unit_testing_controlled/controlled_usb_rosbot_zoomin_move_1/stamped_ground_truth.txt'
vis_odom_file_path = '/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/unit_testing_controlled/controlled_usb_rosbot_zoomin_move_1/stamped_traj_estimate.txt'
data_log_file_path = '/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/unit_testing_controlled/controlled_usb_rosbot_zoomin_move_1/logged_data.txt'

"""
SECTION FOR GROUND TRUTH
"""
# get the ground truth info + plot
gt_data = ExtractData(ground_truth_file_path)
gt_translations = gt_data.translations
gt_quaternions = gt_data.quaternions
# compute the averages for ground truth
gt_avg_translation, gt_avg_euler_angles, gt_euler_angles = get_averages(translations=gt_translations, quaternions=gt_quaternions)
# compute the standard deviations for ground truth
gt_std_translation, gt_std_euler_angles = get_stddevs(translations=gt_translations, quaternions=gt_quaternions)
print("ground truth average translation: {}".format(gt_avg_translation))
print("ground truth translation standard deviation: {}".format(gt_std_translation))
print("ground truth average euler {}".format(gt_avg_euler_angles))
print("ground truth euler standard deviation {}".format(gt_std_euler_angles))


"""
SECTION FOR VISUAL ODOMETRY
"""
# get the ground truth info + plot
vo_data = ExtractData(vis_odom_file_path)
vo_translations = vo_data.translations
vo_quaternions = vo_data.quaternions
# compute the averages for ground truth
vo_avg_translation, vo_avg_euler_angles, vo_euler_angles = get_averages(translations=vo_translations, quaternions=vo_quaternions)
# compute the standard deviations for ground truth
vo_std_translation, vo_std_euler_angles = get_stddevs(translations=vo_translations, quaternions=vo_quaternions)
print("visual odometry average translation: {}".format(vo_avg_translation))
print("visual odometry translation standard deviation: {}".format(vo_std_translation))
print("visual odometry average euler {}".format(vo_avg_euler_angles))
print("visual odometry euler standard deviation {}".format(vo_std_euler_angles))


"""Plotting"""
# plot the ground truth
# plot the ground truth
ground_truth_plot = Plot(translations=gt_translations, euler_angles=gt_euler_angles, plot_destination_path=destination_directory)
ground_truth_plot.plot_and_save("ground_truth")
vis_odom_plot = Plot(translations=vo_translations, euler_angles=vo_euler_angles, plot_destination_path=destination_directory)
vis_odom_plot.plot_and_save("visual_odometry")

""""WRITE THE DATA TO A TXT FILE"""
# Open a text file for writing
with open(data_log_file_path, 'w') as file:
    file.write("ground truth average translation: {}\n".format(gt_avg_translation))
    file.write("ground truth translation standard deviation: {}\n".format(gt_std_translation))
    file.write("ground truth average euler: {}\n".format(gt_avg_euler_angles))
    file.write("ground truth euler standard deviation: {}\n".format(gt_std_euler_angles))
    file.write("\n")
    file.write("visual odometry average translation: {}\n".format(vo_avg_translation))
    file.write("visual odometry translation standard deviation: {}\n".format(vo_std_translation))
    file.write("visual odometry average euler: {}\n".format(vo_avg_euler_angles))
    file.write("visual odometry euler standard deviation: {}\n".format(vo_std_euler_angles))
