#!/usr/bin/python

# Author: Ivy Aiwei Zhang
# Last updated: 08-28-2023
# Purpose: a python file that outputs a list of valid pairs of /usb_cam/image_raw and /stag_marker messages from a given ros bag

# ROS node messages
print("extracting valid pairs of image_raw and stag_marker messages from a given ros bag")

import rosbag
import collections

# construct object tuples for use
ValidPair = collections.namedtuple('ValidPair', ['img_msg', 'marker_msg'])
TimeMessagePair = collections.namedtuple('TimeMessagePair', ['timestamp', 'message'])
TimeImageMarker = collections.namedtuple('TimeImageMarker', ['timestamp', 'img_msg', 'marker_msg'])

ids = collections.defaultdict(lambda: ValidPair(None, None))

print("extracting stream of valid messages")
def distribute_bag_messages(bag_file_path):
    """look for valid /usb_cam/image_raw and /stag_markers messages, and append to corresponding maps under their timestamp"""
    image_raw_messages_map = {}
    stag_marker_messages_map = {}
    with rosbag.Bag(bag_file_path, "r") as bag:

        i = 0

        for topic, bag_message, timestamp in bag.read_messages(topics=['/camera_array/cam1/image_raw/compressed','/stag_markers']):
            i += 1
            if "raw" in topic:
                image_raw_messages_map[bag_message.header.stamp.to_sec()] = bag_message
            else:
                if len(bag_message.markers)>0:
                    stag_marker_messages_map[bag_message.header.stamp.to_sec()] = bag_message

    return image_raw_messages_map, stag_marker_messages_map

def find_matching_map_entries(image_messages_map, marker_messages_map):
    """this method sorts both maps according to ascending key (timestamp) values; then only keep messages with the same timestamp found"""

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
    """This method groups an image message with a marker message occurring at the same timestamp, and append to output stream"""
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
    print("valid input stream successfully extracted")
    # print_valid_message_stream(valid_input_stream)
    return sorted_valid_input_stream
