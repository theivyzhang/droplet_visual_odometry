import numpy as np

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
    # print("here is the rotation matrix: {}".format(rotation_matrix))
    quaternion_rep = self.rotation_matrix_to_quaternion(rotation_matrix)
    # print("here is the quaternion representation: {}".format(quaternion_rep))
    return quaternion_rep
