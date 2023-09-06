#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 08-28-2023
# Purpose: a python file that outputs a list of valid pairs of /usb_cam/image_raw and /stag_marker messages from a given ros bag

# ROS node messages
print("extracting valid pairs of image_raw and stag_marker messages from a given ros bag")

import rosbag
import collections
from decimal import Decimal


ValidPair = collections.namedtuple('ValidPair', ['img_msg', 'marker_msg'])
TimeMessagePair = collections.namedtuple('TimeMessagePair', ['timestamp', 'message'])
TimeImageMarker = collections.namedtuple('TimeImageMarker', ['timestamp', 'img_msg', 'marker_msg'])

ids = collections.defaultdict(lambda: ValidPair(None, None))


"""SECTION: OUTPUT A STREAM OF VALID BAG MESSAGES"""
class BagMessage:
    def __init__(self, i, topic, message, timestamp):
        self.index = i
        self.topic = topic
        self.message = message
        self.timestamp = timestamp


def distribute_bag_messages(bag_file_path):
    image_raw_messages_map ={}
    stag_marker_messages_map ={}
    with rosbag.Bag(bag_file_path, "r") as bag:

        i = 0

        for topic, bag_message, timestamp in bag.read_messages(topics=['/usb_cam/image_raw','/stag_markers']):
            i += 1


            if "raw" in topic:
                image_raw_messages_map[bag_message.header.stamp.to_sec()] = bag_message
            else:
                # do something
                stag_marker_messages_map[bag_message.header.stamp.to_sec()] = bag_message

    return image_raw_messages_map, stag_marker_messages_map

def find_matching_map_entries(image_messages_map, marker_messages_map):
    # Get the common keys between map_a and map_b
    common_keys = set(image_messages_map.keys()) & set(marker_messages_map.keys())

    # Create new filtered dictionaries
    filtered_image_messages_map = {key: value for key, value in image_messages_map.items() if key in common_keys}
    filtered_marker_messages_map = {key: value for key, value in marker_messages_map.items() if key in common_keys}

    # sort the maps again for sanity check
    sorted_filtered_image_messages_map = {k: filtered_image_messages_map[k] for k in sorted(filtered_image_messages_map.keys())}
    sorted_filtered_marker_messages_map ={k: filtered_marker_messages_map[k] for k in sorted(filtered_marker_messages_map.keys())}

    return sorted_filtered_image_messages_map, sorted_filtered_marker_messages_map


def process_stream(sorted_filtered_image_messages_map, sorted_filtered_marker_messages_map):
    # create object [timestamp, image message, marker message]
    assert len(sorted_filtered_image_messages_map) == len(sorted_filtered_marker_messages_map)

    valid_input_stream = []
    for timestamp in sorted_filtered_image_messages_map:
        img_msg = sorted_filtered_image_messages_map[timestamp]
        marker_msg = sorted_filtered_marker_messages_map[timestamp]
        valid_input_stream.append(TimeImageMarker(timestamp=timestamp, img_msg=img_msg, marker_msg=marker_msg))
        # print("{} {}".format(timestamp, marker_msg.markers[0].corners[5].x))

    return valid_input_stream

"""Utility functions"""


def print_valid_message_stream(valid_message_stream):
    i = 0
    for msg in valid_message_stream:
        print("{} {}".format(msg.index, msg.topic))
        i += 1
        if i == 2:
            print("found valid pair!")
            i = 0

def get_valid_message_stream(bag_file_path):
    image_raw_messages_map, stag_marker_messages_map = distribute_bag_messages(bag_file_path)
    sorted_filtered_image_messages_map, sorted_filtered_marker_messages_map = find_matching_map_entries(image_raw_messages_map, stag_marker_messages_map)
    valid_input_stream = process_stream(sorted_filtered_image_messages_map, sorted_filtered_marker_messages_map)
    sorted_valid_input_stream = sorted(valid_input_stream, key=lambda x: x.timestamp)
    print("valid input stream ready")
    # print_valid_message_stream(valid_input_stream)
    print(len(valid_input_stream))
    return sorted_valid_input_stream


# bag_file_path = '/home/ivyz/Documents/ivy_workspace/src/vis_odom/scripts/unit_testing_controlled/controlled_usb_rosbot/forward-3/forward-3.bag'
# get_valid_message_stream(bag_file_path)
