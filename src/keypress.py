#!/usr/bin/env python
import rospy
from hand_direction.msg import action_agent
from getch import getch, pause

# # define hotkeys
# escape = '\x1b'
# exit = '\x03'


def keyboardPublisher():
	key = readKeyboard()
	if key == "w":
		return 1
	elif key == "s":
		return -1
	else:
		return 0

def readKeyboard():
	key = getch()
	if key == "w" or key == "s":
		return key
	else:
		rospy.signal_shutdown("Exit Key")



def talker():
	pub = rospy.Publisher('/rl/action_agent', action_agent)
	rospy.init_node('keyboard')
	while not rospy.is_shutdown():
		act = action_agent()
		act.action = keyboardPublisher()
		pub.publish(act)

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException: pass