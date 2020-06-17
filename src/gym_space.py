#!/usr/bin/env python
import rospy
import gym
from std_msgs.msg import Int16
from main import Game 

class controller:

	def __init__(self):
		print("init")
		self.human_sub = rospy.Subscriber("/RW_x_direction", Int16, self.simulate)
		self.move_rate = 0.1

		self.game = Game()
		

	def simulate(self, x_data):
		if self.game.running:
			self.game.play(x_data.data)
		else:
			self.game.endGame()


if __name__ == '__main__':
	rospy.init_node('hand_direction', anonymous=True)
	ctrl = controller()
	# while ctrl.game.running:
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

	ctrl.env.close()