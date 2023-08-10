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

# GLOBAL VARIABLES
DEFAULT_BASE_LINK_TOPIC = '/base_link'
DEFAULT_CAMERA_TOPIC = '/cam_0_optical_frame'

class GroundTruth:
    def __init__(self, default_base_link_topic=DEFAULT_BASE_LINK_TOPIC, default_camera_topic=DEFAULT_CAMERA_TOPIC):

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
    def get_base_to_marker_homogenous_transformation(self, marker):
        # extract translation and position
        bTm_translation = marker.pose.pose.position
        bTm_orientation = marker.pose.pose.orientation

        # make numpy arrays
        bTm_translation_array = np.array([bTm_translation.x, bTm_translation.y, bTm_translation.z])
        bTm_quaternion_array = np.array([bTm_orientation.x, bTm_orientation.y, bTm_orientation.z, bTm_orientation.w])

        # turn into translation and orientation matrices; dimensions = 4x4
        bTm_translation_mat = tf.transformations.translation_matrix(bTm_translation_array)
        bTm_orientation_mat = tf.transformations.quaternion_matrix(bTm_quaternion_array)

        btm_homogenous_transformation_mat = tf.transformations.concatenate_matrices(bTm_translation_mat,
                                                                                    bTm_orientation_mat)
        return btm_homogenous_transformation_mat # returns 4x4 homogenous transformation matrix

    # function to get the camera to base homogenous transformation matrix
    def get_camera_to_base_homogenous_transformation_matrix(self, marker):
        # using lookupTransform(target, source, time) -> position, quaternion
        cTb_translation, cTb_quaternion = self.listener.lookupTransform(target_frame=self.default_base_link_topic,
                                                                        source_frame=self.default_camera_topic,
                                                                        time=marker.header.stamp)

        # turn into translation and orientation matrices; dimensions = 4x4
        cTb_translation_mat = tf.transformations.translation_matrix(cTb_translation)
        cTb_orientation_mat = tf.transformations.quaternion_matrix(cTb_quaternion)

        cTb_homogenous_transformation_mat = tf.transformations.concatenate_matrices(cTb_translation_mat,
                                                                                    cTb_orientation_mat)
        return cTb_homogenous_transformation_mat

    # this method computes the camera to marker homogenous transformation matrix for a frame; marker = marker.0
    def compute_frame_camera_to_marker(self, marker):

        cTb_homogenous_transformation_mat = self.get_camera_to_base_homogenous_transformation_matrix(marker)

        bTm_homogenous_transformation_mat = self.get_base_to_marker_homogenous_transformation(marker)

        # TODO: checked --> camera2marker = camera2base @ base2marker
        cam_to_marker_transformation = np.matmul(cTb_homogenous_transformation_mat, bTm_homogenous_transformation_mat)

        self.ground_truth_list_cam_to_marker.append(cam_to_marker_transformation)

        return cam_to_marker_transformation # output: 4x4 homogenous transformation matrix

    def get_ground_truth_estimate(self, marker_reading, reference_id=0):
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
                print("marker {} detected!".format(reference_id_index))
                # compute camera to marker transformation for current frame for marker 0
                camera_to_marker_transformation = self.compute_frame_camera_to_marker(markers[reference_id_index])
                return camera_to_marker_transformation


# create the name function
if __name__ == '__main__':
    pass
