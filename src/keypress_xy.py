#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
# from getch import getch, pause
from collaborative_games.msg import observation, action_agent, reward_observation, action_human, action_msg
import std_msgs
import sys, os, termios, fcntl
import time
# from statistics import mean 


offset = 1

UP = 'w'
DOWN = 's'
LEFT = 'a'
RIGHT = 'd'
# LEFT = 'p'
# RIGHT = 'o'

class KeyboardPublisher:

	def __init__(self):
		self.start_time = 0
		self.total_times = []
		self.pub_y = rospy.Publisher('/rl/action_y', action_msg, queue_size=10)
		self.pub_x = rospy.Publisher('/rl/action_x', action_msg, queue_size=10)

	def keyboardPublisher(self, pub_y, pub_x):
		""" 
		Publishes an indicative number for each stroke.
			UP	 -> +1
			Down -> -1
			Left -> +2
			Right-> -2
		"""
		key = self.readKeyboard(pub_y, pub_x)
		if key == UP:
			return 1.
		elif key == DOWN:
			return -1.
		elif key == RIGHT:
			return 2.
		elif key == LEFT:
			return -2.
		else:
			return 0.

	def readKeyboard(self, pub_y, pub_x):
		""" Returns the key if it is of interest."""
		key = self.myGetch(pub_y, pub_x)
		if key == UP or key == DOWN or key == LEFT or key == RIGHT:
			return key
		elif key == "q":
			rospy.signal_shutdown("Exit Key")

	def myGetch(self, pub_y, pub_x):
		""" Reads the keystroke from the keyboard."""
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
				# if counter == 1000:
				# 	h.stamp = rospy.Time.now() 
				# 	act.action = 0
				# 	act.header = h
				# 	self.pub_x.publish(act)
				# 	self.pub_y.publish(act)
				# 	counter = 0
				# counter += 1


		finally:
			termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
			fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

		self.start_time = rospy.get_rostime().to_sec()
		return c



	def talker(self):
		action = self.keyboardPublisher(self.pub_y, self.pub_x)

		h = std_msgs.msg.Header()
		act = action_human()
		h.stamp = rospy.Time.now() 
		act.header = h

		if action == 1 or action == -1:
			act.action = action * offset
			self.pub_y.publish(act)
			self.total_times.append(rospy.get_rostime().to_sec()-self.start_time)
		elif action == 2 or action == -2:
			act.action = action/2 * offset
			self.pub_x.publish(act)

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
