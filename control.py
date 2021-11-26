#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError

link1_length=4.0
link2_length=3.2
link3_length=2.8
z_error=0.4

x_error=-0.14
y_error=0.06

z_error1=0.4
z_error2=0.3
z_error3=0.1


class image_converter:

    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.image_pub1 = rospy.Publisher("image_topic1", Image, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)
        # initialize a publisher to send images from camera2 to a topic named image_topic2
        self.image_pub2 = rospy.Publisher("image_topic2", Image, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback2)
        # initialize a publisher to send robot end-effector position末端执行器位置
        self.end_effector_pub = rospy.Publisher("end_effector_prediction", Float64MultiArray, queue_size=10)
        # initialize a publisher to send desired trajectory轨迹
        self.trajectory_pub = rospy.Publisher("trajectory", Float64MultiArray, queue_size=10)
        # initialize a publisher to send joints' angular position to a topic called joints_pos
        self.joint1_pub = rospy.Publisher("joint_angle_1", Float64, queue_size=10)
        self.joint3_pub = rospy.Publisher("joint_angle_3", Float64, queue_size=10)
        self.joint4_pub = rospy.Publisher("joint_angle_4", Float64, queue_size=10)
        self.target_pos_pub = rospy.Publisher("target_pos", Float64MultiArray, queue_size=10)

        # initialize a publisher to send joints' angular position to the robot
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)

        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        rate = rospy.Rate(50)  # 50hz

        # record the begining time
        self.t0 = rospy.get_time()
        # initialize step time
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')  
        self.time_previous_step2 = np.array([rospy.get_time()], dtype='float64')
        # initialize error and derivative of error for trajectory tracking
        self.error = np.array([0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0], dtype='float64')

        self.b0 = 0
        self.b1 = 0
        self.b2 = 0

        # Detecting the centre of the green circle
    def detect_green(self, image):
        mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

        # Detecting the centre of the yellow circle
    def detect_yellow(self, image):
        mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

        # Detecting the centre of the blue circle
    def detect_blue(self, image):
        mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        if (M['m00'] == 0):
            c = self.detect_yellow(image)
            cx = c[0]
            cy = c[1]
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

        # Detecting the centre of the red circle
    def detect_red(self, image):
        # isolate the blue colour in the image as a binary image
        mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
        # apply a dilate that makes the binary region larger (the more iterations the larger it becomes)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        # obtain the moments of the binary image
        M = cv2.moments(mask)
        # calculate pixel coordinates for the centre of the blob
        if (M['m00'] == 0):
            c = self.detect_blue(image)
            cx = c[0]
            cy = c[1]
        else:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

        # Calculate the conversion from pixel to meter
    def pixel2meter(self, image):
        # Obtain the centre of each coloured blob
        center = self.detect_green(image)
        circle1Pos = self.detect_yellow(image)
        # find the distance between two circles
        dist = np.sum((circle1Pos - center) ** 2)
        return 4 / np.sqrt(dist)

        # camera 1
    def detect_y(self, image):
        a = self.pixel2meter(image)
        # Obtain the centre of each coloured blob
        center = a * self.detect_green(image)
        circle1Pos = a * self.detect_yellow(image)
        circle2Pos = a * self.detect_blue(image)
        circle3Pos = a * self.detect_red(image)
        self.yellow_y = circle1Pos[0] - center[0]
        self.blue_y = circle2Pos[0] - center[0]
        self.red_y = circle3Pos[0] - center[0]
        return np.array([self.yellow_y, self.blue_y, self.red_y])

        # camera 1 or 2
    def detect_z(self, image):
        a = self.pixel2meter(image)
        # Obtain the centre of each coloured blob
        center = a * self.detect_green(image)
        circle1Pos = a * self.detect_yellow(image)
        circle2Pos = a * self.detect_blue(image)
        circle3Pos = a * self.detect_red(image)
        self.yellow_z = -(circle1Pos[1] - center[1])
        self.blue_z = -(circle2Pos[1] - center[1])
        self.red_z = -(circle3Pos[1] - center[1])
        return np.array([self.yellow_z, self.blue_z, self.red_z])

        # camera 2
    def detect_x(self, image):
        a = self.pixel2meter(image)
        # Obtain the centre of each coloured blob
        center = a * self.detect_green(image)
        circle1Pos = a * self.detect_yellow(image)
        circle2Pos = a * self.detect_blue(image)
        circle3Pos = a * self.detect_red(image)
        self.yellow_x = circle1Pos[0] - center[0]
        self.blue_x = circle2Pos[0] - center[0]
        self.red_x = circle3Pos[0] - center[0]
        return np.array([self.yellow_x, self.blue_x, self.red_x])

    def detect_joint_angles(self, image1, image2):
        x = self.detect_x(image2)
        #print(x)
        y = self.detect_y(image1)
        #print(y)
        z = self.detect_z(image1)
        #print(z)

        x = x + x_error
        y = y + y_error

        if (y[1] < 0.01 and y[1] > -0.01):
            y[1] = 0.001

        if (z[1] > 2):
            z[1] = z[1] + z_error1
        elif (z[1] > 1.5):
            z[1] = z[1] + z_error2
        else:
            z[1] = z[1] + z_error3

        if (z[2] > 2):
            z[2] = z[2] + z_error1
        elif (z[2] > 1.5):
            z[2] = z[2] + z_error2
        else:
            z[2] = z[2] + z_error3

        ja1 = np.arctan2(x[1], -y[1])

        if (ja1 < 0.1 and ja1 > -0.1):
            ja1 = 0.0

        # ja3=np.arccos((z[1]-link1_length)/link2_length)
        ja3 = np.arctan(-(-np.sin(ja1) * x[1] + np.cos(ja1) * y[1]) / (z[1] - link1_length))

        if (ja3 < 0.1 and ja3 > -0.1):
            ja3 = 0.0

        # ja4=np.arccos((((z[2]-link1_length)/np.cos(ja3))-link2_length)/link3_length)
        ja4 = np.arctan((+np.cos(ja1) * x[2] + np.sin(ja1) * y[2]) / (
                    +np.sin(ja3) * np.sin(ja1) * x[2] - np.sin(ja3) * np.cos(ja1) * y[2] + np.cos(ja3) * (
                        z[2] - link1_length) - (link2_length + z_error)))

        if (ja4 < 0.1 and ja4 > -0.1):
            ja4 = 0.0

        #print(ja1, ja3, ja4)
        return np.array([ja1, ja3, ja4])

    # Define a circular trajectory
    def trajectory(self):
        cur_time = np.array([rospy.get_time()]) - self.t0
        # y_d = float(6 + np.absolute(1.5* np.sin(cur_time * np.pi/100)))
        self.tx = float(3.0 * np.cos(cur_time * np.pi / 20))
        self.ty = float(4.0 * np.sin(cur_time * np.pi / 14) + 0.5)
        self.tz = float(1.0 * np.sin(cur_time * np.pi / 18) + 4.5)
        return np.array([self.tx, self.ty,self.tz])

    # Calculate the forward kinematics
    def forward_kinematics(self,image1,image2):
        joints = self.detect_joint_angles(image1,image2)
        end_effector = np.array([link3_length*np.cos(joints[0])*np.sin(joints[2])+np.sin(joints[0])*np.sin(joints[1])*(link3_length*np.cos(joints[2])+link2_length),
                                 link3_length*np.sin(joints[0])*np.sin(joints[2])-np.cos(joints[0])*np.sin(joints[1])*(link3_length*np.cos(joints[2])+link2_length),
                                 np.cos(joints[1])*(link3_length*np.cos(joints[2])+link2_length)+link1_length])
        return end_effector

    # Calculate the robot Jacobian, taking derivative of forward kinematics
    def calculate_jacobian(self, image1,image2):
        joints = self.detect_joint_angles(image1,image2)
        jacobian = np.array([[-link3_length*np.sin(joints[0])*np.sin(joints[2])+np.cos(joints[0])*np.sin(joints[1])*(link3_length*np.cos(joints[2])+link2_length),np.sin(joints[0])*np.cos(joints[1])*(link3_length*np.cos(joints[2])+link2_length),link3_length*np.cos(joints[0])*np.cos(joints[2])+np.sin(joints[0])*np.sin(joints[1])*(-link3_length*np.cos(joints[2]))],
                             [link3_length*np.cos(joints[0])*np.sin(joints[2])+np.sin(joints[0])*np.sin(joints[1])*(link3_length*np.cos(joints[2])+link2_length),-np.cos(joints[0])*np.cos(joints[1])*(link3_length*np.cos(joints[2])+link2_length),link3_length*np.sin(joints[0])*np.cos(joints[2])-np.cos(joints[0])*np.sin(joints[1])*(-link3_length*np.cos(joints[2]))],
                             [0,np.sin(joints[1])*(link3_length*np.cos(joints[2])+link2_length),(-link3_length*np.cos(joints[2]))]])
        return jacobian

    # Estimate control inputs for open-loop control 
    def control_open(self, image1, image2):
        # estimate time step
        cur_time = rospy.get_time()
        dt = cur_time - self.time_previous_step2
        self.time_previous_step2 = cur_time
        q = self.detect_joint_angles(image1,image2)  # estimate initial value of joints' 
        print(q)
        J_inv = np.linalg.pinv(self.calculate_jacobian(image1,image2))  # calculating the psudeo inverse of Jacobian
        print(J_inv)
        pos = np.array([self.red_x,self.red_y,self.red_z],dtype="float64")
        print(pos)
        # desired trajectory
        pos_d = self.trajectory()
        print(pos_d)
        # estimate derivative of desired trajectory
        self.error = (pos_d - pos) / dt
        print(self.error)
        # desired joint angles to follow the trajectory
        q_d = q + (dt * np.dot(J_inv, self.error.transpose()))
        print(q_d)
        return q_d

        # Recieve data from camera 1, process it, and publish
    def callback1(self, data):
        # Recieve the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Save the image
        cv2.imwrite('image1.png', self.cv_image1)

        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
        except CvBridgeError as e:
            print(e)

        # Recieve data from camera2, process it, and publish

    def callback2(self, data):
        # Recieve the image
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # Save the image
        cv2.imwrite('image2.png', self.cv_image2)

        a = self.detect_joint_angles(self.cv_image1, self.cv_image2)

        if ((self.b0 > 1 and a[0] < -1) or (self.b0 < -1 and a[0] > 1)):
            a[0] = -a[0]
        self.b0 = a[0]

        if ((self.b1 > 0.5 and a[1] < -0.5) or (self.b1 < -0.5 and a[1] > 0.5)):
            a[1] = -a[1]
        self.b1 = a[1]

        # if((np.abs(self.b2)>0.3 and np.abs(a[2]-self.b2)<0.3) and a[2]*self.b2<0):
        #  a[2]=-a[2]
        # self.b2 = a[2]

        if ((self.b2 > 0.5 and a[2] < -0.5) or (self.b2 < -0.5 and a[2] > 0.5)):
            a[2] = -a[2]
        self.b2 = a[2]

        self.joint_angle_1 = Float64()
        self.joint_angle_1.data = a[0]
        self.joint_angle_3 = Float64()
        self.joint_angle_3.data = a[1]
        self.joint_angle_4 = Float64()
        self.joint_angle_4.data = a[2]

        # send control commands to joints
        q_d = self.control_open(self.cv_image1,self.cv_image2)
        self.joint1 = Float64()
        self.joint1.data = q_d[0]
        self.joint3 = Float64()
        self.joint3.data = q_d[1]
        self.joint4 = Float64()
        self.joint4.data = q_d[2]

        # compare the estimated position of robot end-effector calculated from images with forward kinematics
        x_e = self.forward_kinematics(self.cv_image1, self.cv_image2)
        x_e_image = np.array([self.red_x, self.red_y, self.red_z])
        self.end_effector = Float64MultiArray()
        self.end_effector.data = x_e_image

        # Publishing the desired trajectory on a topic named trajectory
        x_d = self.trajectory()  # getting the desired trajectory
        self.trajectory_desired = Float64MultiArray()
        self.trajectory_desired.data = x_d

        # Publish the results
        try:
            self.robot_joint1_pub.publish(self.joint1)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)
            self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))
            self.joint1_pub.publish(self.joint_angle_1)
            self.joint3_pub.publish(self.joint_angle_3)
            self.joint4_pub.publish(self.joint_angle_4)
            self.end_effector_pub.publish(self.end_effector)
            self.trajectory_pub.publish(self.trajectory_desired)
        except CvBridgeError as e:
            print(e)

# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)

