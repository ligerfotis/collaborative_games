# Collaborative Games
## RoboSKEL, NCSR Demokritos

For Questions, Comments, Clarifications, Suggestions or Discussion:

Fotios Lygerakis (ligerfotis@gmail.com)

-----------------------------------------

The code for this project has been based on this [Github directory](https://github.com/jonastjoms/HumanRobotCollab).

### Description

The purpose of this project is to replicate the [Real-World Human-Robot Collaborative Reinforcement Learning](https://www.researchgate.net/publication/339675313_Real-World_Human-Robot_Collaborative_Reinforcement_Learning) paper. 
Towards this direction we have create a simple graphical maze game with a turtle player we call the "Turtle Game".

The implementation is wrapped in ROS nodes.

The turtle can move in 2 degrees of  freedom (DOF). 
The instances we are studying are:
* 1 DOF is controlled by a human and the other by a RL Agent.
* Both DOF are controlled by two separate humans.

A human user can control a DOF either by keyboard or by moving his/her hand (position of the wrist) on a work station.
For the later this [3D keypoint extractor](https://github.com/ThanasisTs/openpose_utils) is being used.

#### Action Types
There are three ways to give an action to the turtle game:
* Control the Direction of the turtle's acceleration (left or right). The value of the acceleration is constant and predefined.
    `keypoint_to_action_direction_acceleration.py`
* Control both the Direction and the Value of the acceleration. The value of the acceleration is mapped to take values in [-1, 1].
    `keypoint_to_action_acceleration.py`
* Control the velocity of the turtle (velocity = constant  * ( hand_position - current_turtle_position). hand_position is taken from the position of the human's wrist.
    `keypoint_to_action_position.py`


### Experiment
For Human - Agent Collaboration:

        roslaunch play_game_experiment_with_agent.launch
        
For Human - Human Collaboration (Probably does not work properly and needs updates):

        roslaunch play_game_experiment_with_human.launch

In the above launch files the user can select the appropriate action type, by uncommenting the relevant nodes.

### Statistics and plotting

Install dependencies:
In a virtual env

        pip install -r requirements.txt
        
Then run:
    
    python utils.py <argument 1> ...
    
For a detailed description of arguments and functionalities either run it with no arguments:

    python utils
to print the available options or see bellow:


    -trials -filename_mean[path] -filename_stdev[path]
    'Plots the testing score for an experiment.'
    
    -comparison -plot_type[accel, accel_dir] -axis[x, y] 
    'Plots subfigure of  the turtle's position, velocity, acceleration and real actions given, on the same timeline 
        for a specifix axis(x or y). accel, accel_dir are for controlling both value and direction of acceleration or only 
        direction respectively.'

    -plot_x -plot_type[accel, accel_dir] 
    'Plots position, velocity, acceleration and actions on the x axis(human).'
    
    -plot_rl -plot_type[accel, accel_dir] -exp_num 
    'Plots of all the reinforcement learning relevant information of the exp_num experiment with plot_type'
    
    -plot_actions -path -user[agent or human] 
    'Plots user's actions.'
    
    -plot_agent_diff -path 
    'plots boxplot agent's action differences'
    
    -plot_actions_boxplot -path -user[agent or human]
    'plots boxplot of user's actions '
    
    -plot_hist_acts -path
    'Histogram plot of the human and agent actions.'

    -plot_human_act_comparison -path
    'Plot of the human's actions happened and actually used in the game throughout the experiment.'

    -plot_hist_human_act_delay -path
    'Histogram plot of path/human_action_delay_list.csv in the 'path' directory.'
    


