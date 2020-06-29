#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
from getch import getch, pause

# # define hotkeys
# escape = '\x1b'
# exit = '\x03'


def keyboardPublisher():
	key = readKeyboard()
	if key == "w":
		return 1.
	elif key == "s":
		return -1.
	elif key == "d":
		return 2.
	elif key == "a":
		return -2.
	else:
		return 0.

def readKeyboard():
	key = getch()
	if key == "w" or key == "s" or key == "d" or key == "a":
		return key
	elif key == "q":
		rospy.signal_shutdown("Exit Key")



def talker():
	pub_y = rospy.Publisher('/rl/action_y', Float32, queue_size =10)
	pub_x = rospy.Publisher('/rl/action_x', Float32, queue_size =10)

	rospy.init_node('keyboard_y')
	while not rospy.is_shutdown():
		action = keyboardPublisher()
		if action == 1 or action == -1:
			pub_y.publish(action)
		elif action == 2 or action == -2:
			pub_x.publish(action/2)
		else:
			break

if __name__ == '__main__':
    try:
        talker()
    except KeyboardInterrupt:
		print("Shutting down")