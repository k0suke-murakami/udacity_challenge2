#!/usr/bin/python

from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import rosbag
import cv2
import os
from cv_bridge import CvBridge              # , CvBridgeError

#size = (320 * 3, 240)
#pimg = np.zeros(shape=(320, 240, 3))

# ***** main loop *****
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Udacity SDC Challenge-2 Video viewer')
    parser.add_argument('--dataset', type=str, default="/home/aiwagan/data/udacity-dataset_io_vehicle_2016-10-09-03-47-05_0.bag", help='Dataset/ROS Bag name')
    parser.add_argument('--skip', type=int, default="0", help='skip seconds')
    parser.add_argument('--outdir', type=str, default="~/dataset/", help='Provide a specific folder for storing images.')

    args = parser.parse_args()

    dataset = args.dataset
    skip = args.skip
    outdir = args.outdir
    fmt = 'jpg'

    startsec = 0
    bridge = CvBridge()


    # Load the modelng
    #  model= loadsavedmodel(modelfile)
    f_steer = open('steering_angle.csv', 'w')
    f_gps = open('gps.csv', 'w')
    f_cimage = open('center_image.csv', 'w')

    angle_steers = 0
    d = 1
    topics_req = ['/vehicle/steering_report', '/vehicle/gps/fix', '/center_camera/image_color']
    print ("reading rosbag ", dataset)
    bag = rosbag.Bag(dataset, 'r')

    for topic, msg, t in bag.read_messages(topics=topics_req):
        if startsec == 0:
            startsec = t.to_sec()
            if skip < 24 * 60 * 60:
                skipping = t.to_sec() + skip
                print ("skipping ", skip, " seconds from ", startsec, " to ", skipping, " ...")
            else:
                skipping = skip
                print ("skipping to ", skip, " from ", startsec, " ...")
        else:
            if t.to_sec() > skipping:
                if topic in ['/vehicle/steering_report']:
                    f_steer.write(str(msg.header.stamp) + "," + str(msg.steering_wheel_angle)+"\n")

                if topic in ['/vehicle/gps/fix']:
                    f_gps.write(str(msg.header.stamp) + "," + str( msg.latitude) + "," + str(msg.longitude)+"\n")

    f_gps.close()
    f_steer.close()

    df_steer = pd.read_csv('steering_angle.csv')
    df_gps = pd.read_csv('gps.csv')



    vbag = rosbag.Bag('/home/aiwagan/data/udacity-dataset_sensor_camera_center_2016-10-09-03-47-05_0.bag', 'r')
    startsec = 0
    for topic, msg, t in vbag.read_messages(topics=['/center_camera/image_color/compressed']):
        if startsec == 0:
            startsec = t.to_sec()
            if skip < 24 * 60 * 60:
                skipping = t.to_sec() + skip
                print ("skipping ", skip, " seconds from ", startsec, " to ", skipping, " ...")
            else:
                skipping = skip
                print ("skipping to ", skip, " from ", startsec, " ...")
        else:
            if t.to_sec() > skipping:
                if topic in ['/center_camera/image_color/compressed']:
                    f_cimage.write(str(msg.header.stamp) + "\n")



    f_cimage.close()

