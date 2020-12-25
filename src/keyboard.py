#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
# from getch import getch, pause
from collaborative_games.msg import observation, action_agent, reward_observation, action_human, action_msg
import std_msgs
import sys, os, termios, fcntl
import time
from statistics import mean 

offset = 1

UP = 'w'
DOWN = 's'
LEFT = 'a'
RIGHT = 'd'
# LEFT = 'p'
# RIGHT = 'o'

class KeyboardPublisher:

    def __init__(self):
        self.start_time = None
        self.total_times = []
        self.pub_y = rospy.Publisher('/rl/action_y', action_msg, queue_size=2)
        self.pub_x = rospy.Publisher('/rl/action_x', action_msg, queue_size=2)

    def keyboardPublisher(self):
        key = self.readKeyboard()
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

    def readKeyboard(self):
        """ Returns the key if it is of interest."""
        key = self.myGetch()
        if key == UP or key == DOWN or key == LEFT or key == RIGHT:
            return key
        elif key == "q":
            rospy.signal_shutdown("Exit Key")

    def create_msg(self, action):
        # Creates a msg ready to be published from an action
        h = std_msgs.msg.Header()
        msg = action_human()
        h.stamp = rospy.Time.now() 
        msg.header = h
        msg.action = action
        return msg

    def myGetch(self):
        """ Reads the keystroke from the keyboard."""
        h = std_msgs.msg.Header()
        act = action_human()

        fd = sys.stdin.fileno()

        oldterm = termios.tcgetattr(fd)
        newattr = termios.tcgetattr(fd)
        newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, newattr)

        oldf_training_logslags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, oldf_training_logslags | os.O_NONBLOCK)
        counter = 0
        its_time = True
        timer = None
        try:
            while 1:
                if its_time:
                    msg = self.create_msg(0)
                    self.pub_x.publish(msg)
                    self.pub_y.publish(msg)
                    # rospy.sleep(0.5)
                    timer = time.time()
                    its_time = False
                elif (time.time() - timer) > 0.000005:
                    its_time = True
                try:
                    c = sys.stdin.read(1)
                    self.start_time = time.time()
                    its_time - True
                    break
                except IOError: 
                    pass

        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
            fcntl.fcntl(fd, fcntl.F_SETFL, oldf_training_logslags)

        self.start_time = rospy.get_rostime().to_sec()
        return c


    def talker(self):
        action = self.keyboardPublisher()

        if action == 1 or action == -1:
            msg = self.create_msg(action * offset)
            self.pub_y.publish(msg)
            # rospy.sleep(0.05)
            # self.total_times.append(time.time()-self.start_time)
        elif action == 2 or action == -2:
            msg = self.create_msg(action/2 * offset)
            self.pub_x.publish(msg)
            # rospy.sleep(0.05)
        self.total_times.append(time.time()-self.start_time)


if __name__ == '__main__':
    rospy.init_node('keypress_xy', anonymous=True)
    key_pub = KeyboardPublisher()
    try:
        while not rospy.is_shutdown():
            key_pub.talker()
        print("Time from Keystroke to publish is %f milliseconds. \n"%( mean(key_pub.total_times)*1000))
    except KeyboardInterrupt:
        print("Shutting down")
