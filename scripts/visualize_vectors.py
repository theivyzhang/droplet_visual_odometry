# author: Ivy Zhang
# last updated: 07-24-2023
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np



def calculate_unit_vector_ground_truth(ground_truth_vector):
    return ground_truth_vector / np.linalg.norm(ground_truth_vector)


def main():
    # type is np array
    #plt.ion()

    ground_truth_vector = np.array([0.00148313, -0.00242303,  0.001438444])
    print('before convert: ', ground_truth_vector)
    visual_odometry_vector = np.array([0.45204653,  0.7760078 , -0.43984751])
    ground_truth_vector = calculate_unit_vector_ground_truth(ground_truth_vector)
    print('after convert', ground_truth_vector)
    print('vo vector length', np.linalg.norm(visual_odometry_vector))

    vectors_plot = plt.figure()
    vectors_plot_ax = vectors_plot.add_subplot(111, projection='3d')

    # def activate_visualization(self):
    #     self.visualize_ground_truth_vs_visual_odometry(self.ground_truth_vector, self.visual_odometry_vector)


    xs_gt = [0.0, ground_truth_vector[0]]
    ys_gt = [0.0, ground_truth_vector[1]]
    zs_gt = [0.0, ground_truth_vector[2]]

    xs_vo = [0.0, visual_odometry_vector[0]]
    ys_vo = [0.0, visual_odometry_vector[1]]
    zs_vo = [0.0, visual_odometry_vector[2]]

    vectors_plot_ax.plot(xs_gt, ys_gt, zs_gt, color='r', label='ground_truth')
    vectors_plot_ax.plot(xs_vo, ys_vo, zs_vo, color='b', label='visual_odometry')
    print(ground_truth_vector, visual_odometry_vector)

    vectors_plot_ax.set_xlim([-1, 1])
    vectors_plot_ax.set_ylim([-1, 1])
    vectors_plot_ax.set_zlim([-1, 1])

    vectors_plot_ax.set_xlabel('X')
    vectors_plot_ax.set_ylabel('Y')
    vectors_plot_ax.set_zlabel('Z')
    vectors_plot_ax.legend()
    print("got here")
    # plt.title("comparing ground truth and visual odometry")
    plt.show()

#
# # create the name function
if __name__ == '__main__':
    main()


