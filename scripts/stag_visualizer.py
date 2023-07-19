#!/usr/bin/python

import rospy
import tf

from stag_ros.msg import StagMarkers

class StagVisualizer:
    def __init__(self):
        self.br = tf.TransformBroadcaster()
        rospy.Subscriber("bluerov_controller/ar_tag_detector", StagMarkers, self.stag_markers_callback)

    def stag_markers_callback(self, msg):
        rospy.loginfo("publishing " + str(len(msg.markers)))
        for marker in msg.markers:

            self.br.sendTransform((marker.pose.pose.position.x, marker.pose.pose.position.y, marker.pose.pose.position.z),
                (marker.pose.pose.orientation.x, marker.pose.pose.orientation.y, marker.pose.pose.orientation.z, marker.pose.pose.orientation.w),
                msg.header.stamp,
                msg.header.frame_id,
                "marker_{}".format(marker.id)
            )

if __name__ == "__main__":
    rospy.init_node("stag_visualizer")

    s = StagVisualizer()

    rospy.spin()
