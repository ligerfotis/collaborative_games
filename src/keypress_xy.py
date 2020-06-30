#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
# from getch import getch, pause
from hand_direction.msg import observation, action_agent, reward_observation, action_human
import std_msgs
import sys, os, termios, fcntl
import time
from statistics import mean 

# from getch import myGetch
# # define hotkeys
# escape = '\x1b'
# exit = '\x03'
class KeyboardPublisher:

	def __init__(self):
		self.start_time = 0
		self.total_times = []

	def keyboardPublisher(self, pub_y, pub_x):
		key = self.readKeyboard(pub_y, pub_x)
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

	def readKeyboard(self, pub_y, pub_x):
		key = self.myGetch(pub_y, pub_x)
		if key == "w" or key == "s" or key == "d" or key == "a":
			return key
		elif key == "q":
			rospy.signal_shutdown("Exit Key")

	def myGetch(self,pub_y, pub_x):
		h = std_msgs.msg.Header()
		act = action_human()

		fd = sys.stdin.fileno()

		oldterm = termios.tcgetattr(fd)
		newattr = termios.tcgetattr(fd)
		newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
		termios.tcsetattr(fd, termios.TCSANOW, newattr)

		oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
		fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
		counter = 0
		try:        
			while 1:            
				try:
					c = sys.stdin.read(1)
					break
				except IOError: 
					pass
				if counter == 1000:
					h.stamp = rospy.Time.now() 
					act.action = 0
					act.header = h
					pub_x.publish(act)
					pub_y.publish(act)
					counter = 0
				counter += 1


		finally:
			termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
			fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

		self.start_time = rospy.get_rostime().to_sec()
		return c



	def talker(self):
		h = std_msgs.msg.Header()
		

		pub_y = rospy.Publisher('/rl/action_y', action_human, queue_size =10)
		pub_x = rospy.Publisher('/rl/action_x', action_human, queue_size =10)

		act = action_human()


		action = self.keyboardPublisher(pub_y, pub_x)

		h.stamp = rospy.Time.now() 
		act.header = h
		if action == 1 or action == -1:
			act.action = action
			pub_y.publish(act)
			self.total_times.append(rospy.get_rostime().to_sec()-self.start_time)
		elif action == 2 or action == -2:
			act.action = action/2
			pub_x.publish(act)

			self.total_times.append(rospy.get_rostime().to_sec()-self.start_time)


if __name__ == '__main__':
	rospy.init_node('keypress_xy', anonymous=True)
	key_pub = KeyboardPublisher()
	try:
		while not rospy.is_shutdown():
			key_pub.talker()
		print("Time from Keystroke to publish is %f milliseconds. \n"%((mean(key_pub.total_times))*1e3))
	except KeyboardInterrupt:
		print("Shutting down")