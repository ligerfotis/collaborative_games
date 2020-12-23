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

# class _Getch:
#     """Gets a single character from standard input.  Does not echo to the
# screen."""
#     def __init__(self):
#         self.impl = _GetchUnix()

#     def __call__(self): return self.impl()
def create_msg(action):
        h = std_msgs.msg.Header()
        msg = action_human()
        h.stamp = rospy.Time.now() 
        msg.header = h
        msg.action = action
        return msg

def keyboardTranslation(key):
        """ 
        Publishes an indicative number for each stroke.
            UP   -> +1
            Down -> -1
            Left -> +2
            Right-> -2
        """
        if key == UP:
            return 1.
        elif key == DOWN:
            return -1.
        elif key == RIGHT:
            return 2.
        elif key == LEFT:
            return -2.
        elif key == "q":
            rospy.signal_shutdown("Exit Key")
        else:
            return 0.
class _Getch:
    def __init__(self):
        import tty, sys
        self.pub_x = rospy.Publisher('/rl/action_x', action_msg, queue_size=10)
        self.pub_x = rospy.Publisher('/rl/action_x', action_msg, queue_size=10)

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            action = keyboardTranslation(ch)
            if action == 1 or action == -1:
                msg = create_msg(action * offset)
                self.pub_y.publish(msg)
            elif action == 2 or action == -2:
                msg = create_msg(action/2 * offset)
                self.pub_x.publish(msg)
            else:
                msg = self.create_msg(0)
                self.pub_x.publish(msg)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        
        # return ch

getch = _Getch()

class KeyboardPublisher:

    def __init__(self):
        self.start_time = 0
        self.total_times = []
        self.pub_y = rospy.Publisher('/rl/action_y', action_msg, queue_size=10)
        self.pub_x = rospy.Publisher('/rl/action_x', action_msg, queue_size=10)

    def keyboardPublisher(self, pub_y, pub_x):
        """ 
        Publishes an indicative number for each stroke.
            UP   -> +1
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
        key = getch()
        if key == UP or key == DOWN or key == LEFT or key == RIGHT:
            return key
        elif key == "q":
            rospy.signal_shutdown("Exit Key")
        else:
            return 0

    # def myGetch(self, pub_y, pub_x):
    #     """ Reads the keystroke from the keyboard."""
    #     # h = std_msgs.msg.Header()
    #     # act = action_human()

    #     fd = sys.stdin.fileno()

    #     oldterm = termios.tcgetattr(fd)
    #     newattr = termios.tcgetattr(fd)
    #     newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    #     termios.tcsetattr(fd, termios.TCSANOW, newattr)

    #     oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    #     fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
    #     counter = 0
    #     try:
    #         while 1:   
    #             try:
    #                 c = sys.stdin.read(1)
    #                 break
    #             except IOError: 
    #                 pass


    #     finally:
    #         termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
    #         fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)

    #     self.start_time = rospy.get_rostime().to_sec()
    #     return c

    def create_msg(self, action):
        h = std_msgs.msg.Header()
        msg = action_human()
        h.stamp = rospy.Time.now() 
        msg.header = h
        msg.action = action
        return msg

    def talker(self):
        # action = self.keyboardPublisher(self.pub_y, self.pub_x)

        # if action == 1 or action == -1:
        #     msg = self.create_msg(action * offset)
        #     self.pub_y.publish(msg)
        #     self.total_times.append(rospy.get_rostime().to_sec()-self.start_time)
        # elif action == 2 or action == -2:
        #     msg = self.create_msg(action/2 * offset)
        #     self.pub_x.publish(msg)
        #     self.total_times.append(rospy.get_rostime().to_sec()-self.start_time)
        # else:
        #     msg = self.create_msg(0)
        #     self.pub_x.publish(msg)
        getch()

        # rospy.sleep(0.05)


if __name__ == '__main__':
    rospy.init_node('human_keyboad', anonymous=True)
    key_pub = KeyboardPublisher()
    try:
        while not rospy.is_shutdown():
            key_pub.talker()
        print("Time from Keystroke to publish is %f milliseconds. \n"%((mean(key_pub.total_times))*1e3))
    except KeyboardInterrupt:
        print("Shutting down")
