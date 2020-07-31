#!/usr/bin/env python
from math import ceil


human_p = 5
agent_p = 5


height = 800
width = 800

table_max_x = 0.35
table_max_y = 0.35

# should be 12
scale_x = ceil((width/2) / table_max_x)
scale_y = ceil((height/2) / table_max_y)

"""
Converts coordnates x, y from the coord frame being on the top left corner of the game
to the coord frame in the center of the game
"""
def convertCoordinates(x, y):
	x_new = x - (width/2)
	y_new = height - y - (height/2)

	return x_new, y_new

# convert positions from pixels(game) to cm(table) 
def convertPixels2cm(x, y):

	x_px, y_px = convertCoordinates(x, y)

	x_cm = x_px / scale_x
	y_cm = y_px / scale_y

	return x_cm, y_cm

def convertCm2Pixels(x, y):
	x_px = int( x * width / table_max_x)
	y_px = int( y * height / table_max_y)

	return x_px, y_px

# set limits to commands
def regulate(cmd):
	if cmd >= 1:
		return 0.99
	elif cmd <= -1:
		return -0.99
	else:
		return cmd

"""
toDo
"""
def getState_x(robot_state):
	pass

def getState_y(robot_state):
	pass
""""""

def calc_vel_cmd(action, robot_state):
	x, y = action
	x_new, y_new = convertPixels2cm(x, y)

	current_pos_x = getState_x(robot_state)
	current_pos_y = getState_y(robot_state)

	target_pos_x = current_pos_x + x_new
	target_pos_y = current_pos_y + y_new

	human_cmd = regulate(human_p * (target_pos_x - current_pos_x))
	agent_cmd = regulate(agent_p * (target_pos_y - current_pos_y))

	return human_cmd, agent_cmd

   


