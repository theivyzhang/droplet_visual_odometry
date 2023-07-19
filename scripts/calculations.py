import math


def get_3d_euclidean_distance(ground_truth, vis_odom):
    return math.sqrt((vis_odom[0] - ground_truth[0]) ** 2 + (vis_odom[1] - ground_truth[1]) ** 2 + (
            vis_odom[2] - ground_truth[2]) ** 2)


# set1
print(get_3d_euclidean_distance([-3.62774693e-05, 4.90732889e-04, 2.50851359e-04],
                                [5.77350269e-01, -5.77350269e-01, 5.77350269e-01]))  # 1.00015957994

# set2
print(
    get_3d_euclidean_distance([-1.40310731e-04, 5.40626912e-04, -9.02176399e-05], [0.08411758, 0.1224595, 0.98890237]))

# set3
print(get_3d_euclidean_distance([1.13919746, 0.17625815, 0.14474866], [0.00038392, 0.00076788, -0.00388801]))

# set4
print(get_3d_euclidean_distance([0.00013634, -0.00232394,  0.00076512], [-0.04864769, -0.01172476, -0.99874718]))

# set5
print(get_3d_euclidean_distance([4.60664958e-04, 7.28046109e-05, 1.85821428e-04], [0.10878778, 0.01124295, 0.99400142]))
