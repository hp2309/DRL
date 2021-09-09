#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
import csv


def callback(data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    rospy.loginfo(rospy.get_caller_id() + "New Data")
    # file = open('/home/hetal/catkin_ws/src/drl_test/src/log1.txt', 'w')
    # file.write(str(type(data.data))
    # file.write("\n------------------------------------------\n")
    csvfile = open('/home/hetal/catkin_ws/src/drl_test/src/log1.csv', 'w')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(data.data)
    csvfile.close()

# def callback2(data):
#     rospy.loginfo(rospy.get_caller_id() + "New Meta Data")

#     csvfile = open('/home/hetal/catkin_ws/src/drl_test/src/log2.csv', 'w')
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow((data.width, data.height))
#     csvfile.close()
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("map", OccupancyGrid, callback)
    # rospy.Subscriber("/map_metadata", MapMetaData, callback2)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()


384,384