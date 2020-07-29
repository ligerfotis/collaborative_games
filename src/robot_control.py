#!/usr/bin/env python
import tf
from trajectory_execution_msgs.msg import PoseTwist
import geometry_msgs.msg
import rospy
from hand_direction.msg import action_msg

abs_max_theta = 0.1
abs_max_phi = 0.1

class RobotController:

    def __init__(self):
        # Define reset/initial pose:
        self.start_rpy1 = [-0.1, -0.1, 0.0]
        self.start_rpy2 = [0.1, -0.1, 0.0]
        self.start_rpy3 = [-0.1, 0.1, 0.0]
        self.start_pose = tf.transformations.quaternion_from_euler(self.start_rpy1[0], self.start_rpy1[1], self.start_rpy1[2] )
        self.pose_goal = geometry_msgs.msg.Pose()
        self.pose_goal.orientation.x = self.start_pose[0]
        self.pose_goal.orientation.y = self.start_pose[1]
        self.pose_goal.orientation.z = self.start_pose[2]
        self.pose_goal.orientation.w = self.start_pose[3]
        self.pose_goal.position.x = 0.3
        self.pose_goal.position.y = 0.3
        self.pose_goal.position.z = 0.4

        self.current_pose = None
        self.current_rpy = None
        self.agent_action = None

        self.listener = tf.TransformListener()

        self.robot_state_sub = rospy.Subscriber("/manos_cartesian_velocity_controller_sim/ee_state", PoseTwist, self.set_robot_state)
        self.agent_action_sub = rospy.Subscriber("/rl/action_y", action_msg, self.set_agent_action)
        # self.reset_sub = rospy.Subscriber("robot_reset", Int16, self.reset)

        self.cmd = rospy.Publisher('/manos_cartesian_velocity_controller_sim/command_cart_vel', geometry_msgs.msg.Twist, queue_size=100)

    def set_robot_state(self, robot_state):

       self.current_pose = robot_state.pose
       self.current_rpy = robot_state.twist.angular

    def set_agent_action(self, agent_action):
        if agent_action.action != 0.0:
            self.agent_action = agent_action.action

    def reset(self):
        # Initialize publisher and listener
        training_msg = geometry_msgs.msg.Twist()
        try:
            # Setting linear commands to be published
            training_msg.linear.x = self.pose_goal.position.x - self.current_pose.position.x
            training_msg.linear.y = self.pose_goal.position.y - self.current_pose.position.y
            training_msg.linear.z = self.pose_goal.position.z - self.current_pose.position.z
            training_msg.angular.z = 0 - self.current_rpy[2]
            training_msg.angular.x = 0
            training_msg.angular.y = 0
        except:
            pass
        # Publish action
        self.cmd.publish(training_msg)


    def control(self):
        # Initialize publisher and listener
        training_msg = geometry_msgs.msg.Twist()

        try:
            # Setting linear commands to be published
            training_msg.linear.x = self.pose_goal.position.x - self.current_pose.position.x
            training_msg.linear.y =  self.agent_action/10
            # training_msg.linear.y = self.pose_goal.position.y - self.current_pose.position.y
            
            training_msg.linear.z = self.pose_goal.position.z - self.current_pose.position.z
            training_msg.angular.z = 0 - self.current_rpy[2]
            training_msg.angular.x = 0
            training_msg.angular.y = 0
        except:
            pass
        
        # # Setting angular commands, enforcing constraints
        # if abs(current_rpy[0]) > abs_max_theta:
        #     if current_rpy[0] < 0 and total_action[0] < 0:
        #         training_msg.twist.angular.x = 0
        #     elif current_rpy[0] > 0 and total_action[0] > 0:
        #         training_msg.twist.angular.x = 0
        #     else:
        #         training_msg.twist.angular.x = total_action[0]
        # else:
        #     training_msg.twist.angular.x = total_action[0]
        # if abs(current_rpy[1]) > abs_max_phi:
        #     if current_rpy[1] < 0 and total_action[1] < 0:
        #         training_msg.twist.angular.y = 0
        #     elif current_rpy[1] > 0 and total_action[1] > 0:
        #         training_msg.twist.angular.y = 0
        #     else:
        #         training_msg.twist.angular.y = total_action[1]
        # else:
        #     training_msg.twist.angular.y = total_action[1]
        # Need to transform to base_link frame for planning
        # rot_vector = geometry_msgs.msg.Vector3()
        # rot_vector = training_msg.angular
        # try:
        #     rot_vector = self.listener.transformVector3("base_link", rot_vector)
        # except tf.Exception:
        #     ROS_ERROR("%s",ex.what())
        #     rospy.sleep(1)
        #     pass
        # training_msg.angular = rot_vector
        # if training_msg.angular.x > 1:
        #     training_msg.angular.x = 1
        # elif training_msg.angular.x < -1:
        #     training_msg.angular.x = -1
        # if training_msg.angular.y > 1:
        #     training_msg.angular.y = 1
        # elif training_msg.angular.y < -1:
        #     training_msg.angular.y = -1
        if training_msg.linear.x > 1:
            training_msg.linear.x = 1
        elif training_msg.linear.x < -1:
            training_msg.linear.x = -1
        if training_msg.linear.y > 1:
            training_msg.linear.y = 1
        elif training_msg.linear.y < -1:
            training_msg.linear.y = -1

        # Publish action
        self.cmd.publish(training_msg)
        # rospy.sleep(0.2)

if __name__ == '__main__':
    rospy.init_node('robot_controller', anonymous=False)
    ctrl = RobotController()
    ctrl.reset()
    while 1:
        ctrl.control()
    # while ctrl.game.running:
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")