#include <stdio.h>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>
#include <hrc_msgs/observations.h>
#include <tf/transform_datatypes.h>
#include <tf/tf.h>
#include <ros/console.h>
#include <math.h>


int main(int argc, char** argv)
{
  ros::init(argc,argv, "observations");
  ros::NodeHandle node;

// Publisher object and topic it publishes to
  ros::Publisher observations =
  node.advertise<hrc_msgs::observations>("/observations", 10);

// Initializing observation variables
  // Robot`s tray rotation around x with respect to world:
  double robot_theta;
  // Robot`s tray rotation around y with respect to world:
  double robot_phi;
  // Robot`s tray rotation around z with respect to world:
  double robot_psi;
  // Robot`s tray angular velocity around x with respect to world:
  double robot_theta_dot;
  // Robot`s tray angular velocity around y with respect to world:
  double robot_phi_dot;
  // Robot`s tray angular velocity around z with respect to world:
  double robot_psi_dot;
  // Ball position on x axis in tray frame:
  double ball_x;
  // Ball position on y axis in tray frame:
  double ball_y;
  // Ball velocity on x axis in tray frame:
  double ball_vel_x;
  // Ball velocity on y axis in tray frame:
  double ball_vel_y;
  // Human`s tray rotation around x with respect to world:
  double human_theta;
   // Human`s tray rotation around y with respect to world:
  double human_phi;
  // Human`s tray rotation around z with respect to world:
  double human_psi;
  // Human`s tray angular velocity around x with respect to world:
  double human_theta_dot;
  // Human`s tray angular velocity around y with respect to world:
  double human_phi_dot;
  // Human`s tray angular velocity around z with respect to world:
  double human_psi_dot;

  hrc_msgs::observations obs;

// Initialize listener
  tf::TransformListener listener;

// Initialize transforms
  tf::StampedTransform robot_tray_transform;
  tf::StampedTransform human_tray_transform;
  tf::StampedTransform ball_transform;

// Initialize quaternions
  tf::Quaternion robot_tray;
  tf::Quaternion human_tray;



// Wait for transforms
  try{
    ros::Time now = ros::Time::now();
    listener.waitForTransform("/world", "/Human",
                            now, ros::Duration(3.0));
    listener.waitForTransform("/tray", "/Ball",
                            now, ros::Duration(3.0));
    listener.waitForTransform("/world", "/tray",
                            now, ros::Duration(3.0));
  }
  catch (tf::TransformException &ex) {
    ROS_ERROR("%s",ex.what());
    ros::Duration(1.0).sleep();
  }

// Initial ball position for velocity calculations
  try{
    listener.lookupTransform("/tray", "/Ball",
                           ros::Time(0), ball_transform);
     }
  catch (tf::TransformException &ex) {
    ROS_ERROR("%s",ex.what());
    ros::Duration(1.0).sleep();
  }
// Initializing translation of ball in tray frame
  ball_x = ball_transform.getOrigin().x();
  ball_y = ball_transform.getOrigin().y();

// Initializing robot`s tray rotation
  try{
  listener.lookupTransform("/world", "/tray",
                           ros::Time(0), robot_tray_transform);
  listener.lookupTransform("/world", "/Human",
                          ros::Time(0), human_tray_transform);
     }
  catch (tf::TransformException &ex) {
    ROS_ERROR("%s",ex.what());
    ros::Duration(1.0).sleep();
     }
// Extracting rotations
  robot_tray = robot_tray_transform.getRotation();
  human_tray = human_tray_transform.getRotation();
  tf::Matrix3x3 m0(robot_tray);
  m0.getRPY(robot_theta, robot_phi, robot_psi);
  tf::Matrix3x3 m1(human_tray);
  m1.getRPY(human_theta, human_phi, human_psi);
// Initializing rotation variables for velocity calculations
  double new_robot_theta;
  double new_robot_phi;
  double new_robot_psi;
  double new_human_theta;
  double new_human_phi;
  double new_human_psi;


// Start while loop where transforms are extracted
  int r = 30;
  ros::Rate rate(r);
  while (node.ok()){

// Looking up the three needed transforms
    try{
      listener.lookupTransform("/world", "/Human",
                               ros::Time(0), human_tray_transform);
      listener.lookupTransform("/tray", "/Ball",
                               ros::Time(0), ball_transform);
      listener.lookupTransform("/world", "/tray",
                               ros::Time(0), robot_tray_transform);
    }
    catch (tf::TransformException &ex) {
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
      continue;
    }

// Extracting rotations
    human_tray = human_tray_transform.getRotation();
    robot_tray = robot_tray_transform.getRotation();

    tf::Matrix3x3 m2(human_tray);
    m2.getRPY(new_human_theta, new_human_phi, new_human_psi);

    tf::Matrix3x3 m3(robot_tray);
    m3.getRPY(new_robot_theta, new_robot_phi, new_robot_psi);

// Setting angular velocitoes
    robot_theta_dot = (robot_theta - new_robot_theta)*r;
    robot_phi_dot = (robot_phi - new_robot_phi)*r;
    human_theta_dot = (human_theta - new_human_theta)*r;
    human_phi_dot = (human_phi - new_human_phi)*r;

// Update tray rotations
    robot_theta = new_robot_theta;
    robot_phi = new_robot_phi;
    robot_psi = new_robot_psi;
    human_theta = new_human_theta;
    human_phi = new_human_phi;
    human_psi = new_human_psi;

// Setting ball velocity
    ball_vel_x = (ball_x - ball_transform.getOrigin().x())*r;
    ball_vel_y = (ball_y - ball_transform.getOrigin().y())*r;

// Updating translation of ball in tray frame
    ball_x = ball_transform.getOrigin().x();
    ball_y = ball_transform.getOrigin().y();

// Update observation object
    obs.header.stamp = ros::Time::now();
    obs.robot_theta = robot_theta;
    obs.robot_phi = robot_phi;
    obs.robot_psi = robot_psi;
    obs.robot_theta_dot = robot_theta_dot;
    obs.robot_phi_dot = robot_phi_dot;
    obs.ball_x = ball_x;
    obs.ball_y = ball_y;
    obs.ball_vel_x = ball_vel_x;
    obs.ball_vel_y = ball_vel_y;
    obs.human_theta = human_theta;
    obs.human_phi = human_phi;
    obs.human_psi = human_psi;
    obs.human_theta_dot = human_theta_dot;
    obs.human_phi_dot = human_phi_dot;

// Publish
    observations.publish(obs);

    rate.sleep();
  }
  return 0;
};
