# Collaborative Games
## RoboSKEL, NCSR Demokritos

The code for this project has been based on this [Github directory](https://github.com/jonastjoms/HumanRobotCollab).

### Description

The purpose of this project is to replicate the [Real-World Human-Robot Collaborative Reinforcement Learning](https://www.researchgate.net/publication/339675313_Real-World_Human-Robot_Collaborative_Reinforcement_Learning) paper. 
Towards this direction we have create a simple graphical maze game with a turtle player we call the "Turtle Game".

The implementation is wrapped in ROS nodes.

The turtle can move in 2 degrees of  freedom (DOF). 
The instances we are studying are:
* 1 DOF is controlled by a human and the other by a RL Agent.
* Both DOF are controlled by two separate humans.

A human user can control a DOF either by keyboard or by moving his/her hand on a work station.
For the later this [3D keypoint extractor](https://github.com/ThanasisTs/openpose_utils) is being used.

### Code
For Human - Agent Collaboration:

        roslaunch play_game_experiment_with_agent.launch
        
For Human - Human Collaboration:

        roslaunch play_game_experiment_with_human.launch
