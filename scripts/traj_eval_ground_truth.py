#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 8-10-2023
# Purpose: this file extract ground truth from a single marker reading;
# ground truth in the form of camera_to_marker homogenous transformation matrix

# ROS node messages
print("extracting ground truth from two consecutive frames and giving us the marker to marker transformations")

# other packages
import numpy as np
import tf as tf
import itertools
import rospy
import pose_estimation_module as PEM

# GLOBAL VARIABLES
DEFAULT_BASE_LINK_TOPIC = '/base_link'
DEFAULT_CAMERA_TOPIC = '/cam_0_optical_frame'

DEFAULT_CTB_TRANSLATION = [-0.000, -0.000, -0.133]
DEFAULT_CTB_QUATERNION = [0.500, -0.500, 0.500, 0.500]



class GroundTruth:
    def __init__(self, default_base_link_topic=DEFAULT_BASE_LINK_TOPIC, default_camera_topic=DEFAULT_CAMERA_TOPIC,
                 default_ctb_translation=DEFAULT_CTB_TRANSLATION, default_ctb_quaternion=DEFAULT_CTB_QUATERNION):

        """
        by initializing vo, the image subscriber in hypothesis is activated
        uncomment the draw matches section that save the matched image
        """

        # list to store information
        self.ground_truth_full_list_in_base_link = []
        self.ground_truth_list_cam_to_marker = []

        # set up the needed flags
        self.default_base_link_topic = default_base_link_topic
        self.default_camera_topic = default_camera_topic

        # initialize transform listener
        self.listener = tf.TransformListener()
        self.ctb_translation = default_ctb_translation
        self.ctb_quaternion = default_ctb_quaternion
        # rospy.wait_for_service('spawn')

        """Parameters for pixel length testing """
        self.objective_marker_length_doc = "/home/ivyz/Documents/UAV_VisOdom_Data/cart_experiment/data_20231101/counterclockwise_1/objective_marker_length.txt"
        self.biased_marker_length_doc = "/home/ivyz/Documents/UAV_VisOdom_Data/cart_experiment/data_20231101/counterclockwise_1/biased_marker_length.txt"
        PEM.clear_txt_file_contents(self.objective_marker_length_doc)
        PEM.clear_txt_file_contents(self.biased_marker_length_doc)
        self.objective_count = 0
        self.biased_count = 0

    """
    this method returns the quaternion representation of a rotation matrix
    """

    def rotation_matrix_to_quaternion(self, rotation_matrix):
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

    def quaternion_representation(self, homogenous_matrix):
        rotation_matrix = homogenous_matrix[:3, :3]
        quaternion_rep = self.rotation_matrix_to_quaternion(rotation_matrix)
        return quaternion_rep

    """
    these methods computes the camera to marker translation (ground truth) at a given frame
    """

    # function to get the base to marker homogenous transformation matrix

    # TODO: check if camera to marker means -  position of marker relative the camera in cmaera coord frame
    def get_base_ref_frame_to_marker_homogenous_transformation(self, marker):
        # extract translation and position of the marker from the camera reference frame
        bTm_translation = marker.pose.pose.position
        bTm_orientation = marker.pose.pose.orientation

        # make numpy arrays
        bTm_translation_array = np.array([bTm_translation.x, bTm_translation.y, bTm_translation.z])
        bTm_quaternion_array = np.array([bTm_orientation.x, bTm_orientation.y, bTm_orientation.z, bTm_orientation.w])

        # turn into translation and orientation matrices; dimensions = 4x4
        bTm_translation_mat = tf.transformations.translation_matrix(bTm_translation_array)
        bTm_orientation_mat = tf.transformations.quaternion_matrix(bTm_quaternion_array)

        # TODO: checked with Sam - translation then rotation gives the right combination
        bTm_homogenous_transformation_mat = tf.transformations.concatenate_matrices(bTm_translation_mat, bTm_orientation_mat)
        # print("translation in the btm 4x4 matrix= {}".format(bTm_homogenous_transformation_mat[:3, 3]))
        return bTm_homogenous_transformation_mat  # returns 4x4 homogenous transformation matrix

    # function to get the camera to base homogenous transformation matrix
    def get_camera_to_base_homogenous_transformation_matrix(self):

        cTb_translation, cTb_quaternion = self.ctb_translation, self.ctb_quaternion

        # turn into translation and orientation matrices; dimensions = 4x4
        cTb_translation_mat = tf.transformations.translation_matrix(cTb_translation)
        cTb_orientation_mat = tf.transformations.quaternion_matrix(cTb_quaternion)

        cTb_homogenous_transformation_mat = tf.transformations.concatenate_matrices(cTb_translation_mat,
                                                                                    cTb_orientation_mat)
        return cTb_homogenous_transformation_mat

    # this method computes the camera to marker homogenous transformation matrix for a frame; marker = marker.0
    def compute_frame_camera_to_marker(self, marker, base_link_flag=True):

        if base_link_flag:
            cTb_homogenous_transformation_mat = self.get_camera_to_base_homogenous_transformation_matrix()

            bTm_homogenous_transformation_mat = self.get_base_ref_frame_to_marker_homogenous_transformation(marker)

            # TODO: checked --> camera2marker = camera2base @ base2marker
            cam_to_marker_transformation = np.matmul(cTb_homogenous_transformation_mat, bTm_homogenous_transformation_mat)
        else:
            cam_to_marker_transformation = self.get_base_ref_frame_to_marker_homogenous_transformation(marker)

        self.ground_truth_list_cam_to_marker.append(cam_to_marker_transformation)

        return cam_to_marker_transformation  # output: 4x4 homogenous transformation matrix

    # TODO: added new method to retrieve marker length in pixels; subject to finetuning*
    # TODO: can show stag marker message in rqt_bag
    def distance(self, point1, point2):
        # print("point 1 and point 2 are {}, {}".format(point1, point2))
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]
        return np.linalg.norm(np.array([x1, y1])-np.array([x2, y2]))

    def calculate_centroid(self, corners):

        x_values = [corner.x for corner in corners]
        y_values = [corner.y for corner in corners]

        centroid_x = sum(x_values) / len(corners)
        centroid_y = sum(y_values) / len(corners)

        # print("centroid x and y: {}, {}".format(centroid_x, centroid_y))

        return centroid_x, centroid_y

    def reorder_corners(self, corners):
        # assuming the corners are unordered
        # print("corners in reorder corners: {}".format(corners))

        # Convert the list of coordinates to a NumPy array
        corner_array = np.array([[corner.x, corner.y] for corner in corners])

        # Calculate the centroid
        centroid = np.mean(corner_array, axis=0)

        # Calculate the angles of each corner relative to the centroid
        angles = np.arctan2(corner_array[:, 1] - centroid[1], corner_array[:, 0] - centroid[0])

        # sort the corners based on their angles
        sorted_indices = np.argsort(angles)

        # Use the sorted indices to reorder the corners
        ordered_corners = [corners[i] for i in sorted_indices]

        return ordered_corners


    def calculate_side_length(self, corners):
        # Extract x and y coordinates from the _geometry_msgs__Point objects
        x_values = [corner.x for corner in corners]
        y_values = [corner.y for corner in corners]

        # Calculate the side lengths using the coordinates
        side_lengths = [np.linalg.norm([x_values[i] - x_values[(i + 1) % 4], y_values[i] - y_values[(i + 1) % 4]]) for i in range(4)]

        side_length = np.mean(side_lengths)

        return side_length


    def get_current_marker_pixel_length(self, marker_message):
        marker_corners = marker_message.markers[0].corners
        # print out the corners
        if len(marker_corners)>0:
            # for corner in marker_corners:
            #     print("we have corner with x, y as {}, {}".format(corner.x, corner.y))
            print(" --- now calculating the side length --- ")

            # group corners into sets of 4 based on the shortest pairwise distances
            four_corner_sets = []
            remaining_indices = set(range(len(marker_corners)))

            while remaining_indices:
                # take the first remaining index as a starting point
                current_index = remaining_indices.pop()

                # calculate the centroid of the corner sets
                centroid_x, centroid_y = self.calculate_centroid([marker_corners[current_index]])
                current_centroid = (centroid_x, centroid_y)
                current_corner = (marker_corners[current_index].x, marker_corners[current_index].y)

                # find the closest remaining corners to the current corner
                closest_indices = sorted(remaining_indices, key=lambda index:self.distance(current_centroid, current_corner))[:3]

                # add the current index and its three closest neighbors to a set
                corner_set = [current_index] + closest_indices
                four_corner_sets.append(corner_set)

                # remove these four indices from the list of remaining indices
                remaining_indices.difference_update(corner_set)

            # calculate the side lengths for each marker
            side_lengths = []
            for corner_set in four_corner_sets:
                ordered_corners = self.reorder_corners([marker_corners[index] for index in corner_set])
                # print("reordered corner set: {}".format(ordered_corners))

                # calculate the side length of the marker
                side_length = self.calculate_side_length(ordered_corners)

                side_lengths.append(side_length)

            average_side_length = np.mean(side_lengths)
            print("we have average side length {}".format(average_side_length))
            print("--- finished marker pixel length estimation ---")
            with open(self.objective_marker_length_doc, 'a') as file:
                file.write("at {}, biased marker pixel length = {} \n".format(str(self.objective_count), str(average_side_length)))
                self.objective_count+=1

            return average_side_length
        else:
            return 0

    def get_current_marker_pixel_length_2(self, marker_message):
        # print(marker_message.markers[0])
        count = 0
        if len(marker_message.markers[0].corners) > 0:
            # print(type(marker_message.markers[0].corners))
            marker_A_corners_x = [marker_message.markers[0].corners[0].x, marker_message.markers[0].corners[1].x,
                                marker_message.markers[0].corners[2].x, marker_message.markers[0].corners[3].x]

            min_X = min(marker_A_corners_x)
            max_X = max(marker_A_corners_x)

            marker_pixel_length = max_X - min_X
            # print("the marker pixel length is {}".format(marker_pixel_length))
            with open(self.biased_marker_length_doc, 'a') as file:
                file.write("at {}, biased marker pixel length = {} \n".format(str(self.biased_count), str(marker_pixel_length)))
                self.biased_count += 1

        else:
            return 0

    def get_marker_position(self, marker_reading, reference_id, base_link_flag=True):
        # callback function to access the ground truth data
        markers = marker_reading.markers  # get the marker information

        # taking marker 0 information
        if len(markers) > 0:
            reference_id_index = -1
            for i, m in enumerate(markers):
                if m.id == reference_id:
                    reference_id_index = i
                    break

            if reference_id_index == -1:
                return None
            else:
                # print("marker {} detected!".format(reference_id_index))
                # compute camera to marker transformation for current frame for marker 0

                camera_to_marker_transformation = self.compute_frame_camera_to_marker(markers[reference_id_index], base_link_flag)
                return camera_to_marker_transformation



    # TODO: implement the function that returns the list of keypoints produced by the StagMarker Message
    def get_stagmarker_keypoints(self):
        # in pixel coordinates

        # sort if needed; sorting mechanism to be discussed
        pass


# create the name function
if __name__ == '__main__':
    pass
