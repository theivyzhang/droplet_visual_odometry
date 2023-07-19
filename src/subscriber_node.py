# #!/usr/bin/python
# import rospy
# from std_msgs.msg import String
#
# def callback(data):
#     rospy.loginfo("RECEIVED DATA: %s", data.data)
#
#
# def listener():
#     # subscribe to the topic
#     # initalize the node
#     rospy.init_node("subscriber_node", anonymous=True)
#     rospy.Subscriber("/camera_array/cam0/image_raw/compressed", String, callback)
#     rospy.spin()
#
#
#
#
#
# #create the name function
# if __name__ == '__main__':
#     try:
#         listener()
#     except rospy.ROSInterruptException:
#         pass
