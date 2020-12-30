# Maze 3D Collaborative Learning on shared task

Maze 3D game from: https://github.com/amengede/Marble-Maze

Reinforcement Learning (RL) Agent: Soft Actor Critic (SAC)

(work still in progress)

### Prerequisites 
* ROS distribution: melodic
* OS: ubuntu 18.04
* (Recommended) create a python virtual environment

        python3 -m venv env
        source venv/bin/activate
        pip install -r requirements.txt

* `maze3d_ros_wrapper.py` must be executed with the above python 3 environment.
  * substitute `#!<path_to_env>/bin/python` on the top of the file

### Learn the task collaboratively
   
* Adjust the hyperparameters in the `config_sac.yaml` or the `config_human.yaml` file
    * Note: Discrete SAC is only compatible with the game so far
  

* In the `maze3D_SAC_Human_keyboard.launch` choose config file:
    * Human - SAC agent: `args="config_sac.yaml"`
    * Human - Human: `args="config_human.yaml"`
  

* In a terminal run
    
      roslaunch collaborative_games maze3D_SAC_Human_keyboard.launch
 

* Use `a`(left) and `d`(right) keys to control the tilt of the tray around its vertical(y) axis
  * keyboard strokes are being consumed by in `keyboard.py`(in the `maze3D_SAC_Human_keyboard.launch` file)
  * `keyboard.py` publish actions on the `rl/action_<x>` topics.
  

* The goal can be set in the maze3D/utils.py file
    e.g. 
  
        #################
        goal = left_down
        ################




