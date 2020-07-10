#!/usr/bin/env python
from __future__ import print_function

import time
import timeit
from math import floor
import time
import pygame
from pygame.locals import *
from hand_direction.msg import action_agent
import matplotlib.pyplot as plt
import numpy as np
from hyperparams_ur10 import MAX_STEPS
import rospkg

accel_rate_x = 1 * 1e-3
accel_rate_y = 5 * 1e-3


backgroundColor = (255, 255, 255)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
bright_red = (255, 0, 0)
bright_green = (0, 255, 0)

# path = "/home/fligerakis/catkin_ws/src/hand_direction/src/"
rospack = rospkg.RosPack()
package_path = rospack.get_path("hand_direction")

def text_objects(text, font):
    textSurface = font.render(text, True, BLACK)
    return textSurface, textSurface.get_rect()


def quit_game():
    pygame.display.quit()
    pygame.quit()
    exit(0)

class Game:
    def __init__(self):
        self.experiment = 0
        self.TIME = 10
        self.start_time = None
        self.time_elapsed = 0
        # 2 - Initialize the game
        pygame.init()
        self.width, self.height = 800, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        flags = DOUBLEBUF
        self.screen = pygame.display.set_mode((self.width, self.height), flags)
        self.screen.set_alpha(None)

        self.keys = [False, False, False, False]
        self.turtle_pos = [5, self.height - 64]
        self.reward = 0
        # 3 - Load images
        self.player = pygame.image.load(package_path + "/src/turtle.png").convert_alpha()
        self.youwin = pygame.image.load(package_path + "/src/youwin.png").convert_alpha()
        obst_x, obsty = self.width / 2, self.height / 2
        self.running = 1
        self.exitcode = 0
        # obstacle dimensions and position
        self.thing_startx = self.width / 4
        self.thing_starty = self.height / 2
        self.thing_width = 400
        self.thing_height = 30
        self.player_width = 64

        self.vel_y = self.vel_x = 0

        self.point_1a = (60 - 50, self.height - 60 - 50)
        self.point_2a = (self.width - 60 - 50, 60 - 50)
        self.point_1b = (60 + 50, self.height - 60 + 50)
        self.point_2b = (self.width - 60 + 50, 60 + 50)

        self.timedOut = self.finished = False

        self.limit1 = self.width / 2
        self.limit2 = self.width / 2 - 100

        self.accel_x_list = []
        self.accel_y_list = []
        self.vel_x_list = []
        self.vel_y_list = []
        self.time = []
        self.count = 0
        self.global_start_time = time.time()
        self.turtle_real_x_pos_list = []
        self.turtle_real_y_pos_list = []

        self.accel_x, self.accel_y, self.vel_x, self.vel_y = [0, 0, 0, 0]

        self.clock = pygame.time.Clock()
        self.time_dependend = True

        self.intro = True
        self.pause = False

        pygame.display.update()

    def obstacle(self):

        pygame.draw.line(self.screen, BLACK, self.point_1a, self.point_2a)
        pygame.draw.line(self.screen, BLACK, self.point_1b, self.point_2b)


    def obstacle2(self):

        pygame.draw.line(self.screen, BLACK, [0, 0], [self.limit1, self.limit1])
        pygame.draw.line(self.screen, BLACK, [self.limit2, self.limit2], [self.width,self.height])


    def obstacle3(self):

        pygame.draw.line(self.screen, BLACK, [0, self.height / 2], [self.limit2, self.height / 2])
        pygame.draw.line(self.screen, BLACK, [self.limit1, self.height / 2], [self.width, self.height / 2])


    def button(self, msg, x, y, w, h, ic, ac, action=None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        # print(click)
        if x + w > mouse[0] > x and y + h > mouse[1] > y:
            pygame.draw.rect(self.screen, ac, (x, y, w, h))
            if click[0] == 1 and action is not None:
                if action == "play":
                    self.start_time = time.time()
                    self.intro = False
                elif action == "quit":
                    quit_game()
                elif action == "reset":
                    self.turtle_pos = [5, self.height - 64]
                elif action == "unpause":
                    self.pause = False
                elif action == "pause":
                    self.pause = True
                    self.paused()
        else:
            pygame.draw.rect(self.screen, ic, (x, y, w, h))

        pygame.font.init()
        smallText = pygame.font.SysFont("comicsansms", 20)
        textSurf, textRect = text_objects(msg, smallText)
        textRect.center = ((x + (w / 2)), (y + (h / 2)))
        self.screen.blit(textSurf, textRect)


    def game_intro(self):
        while self.intro:
            for event in pygame.event.get():
                # print(event)
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            self.screen.fill(WHITE)
            pygame.font.init()
            largeText = pygame.font.Font(None, 115)
            TextSurf, TextRect = text_objects("Turtle Collab", largeText)
            TextRect.center = ((self.width / 2), (self.height / 2))
            self.screen.blit(TextSurf, TextRect)

            try:
                self.button("GO!", 150, 450, 100, 50, GREEN, bright_green, "play")
                self.button("Quit", 550, 450, 100, 50, RED, bright_red, "quit")
            except pygame.error:
                print("An exception occurred")

            pygame.display.update()
            self.clock.tick(15)

    def play(self, data=None):
        # print(data)
        start_time = time.time()
        if data is None:
            data = [0, 0]

        x_data = data[0]
        y_data = data[1]

        # 5 - clear the screen before drawing it again
        self.screen.fill(backgroundColor)
        # 6 - draw the screen elements
        pygame.draw.circle(self.screen, RED, (self.width - 80, 80), 60)  # up-right

        self.obstacle3()
        turtle = self.screen.blit(self.player, self.turtle_pos)
        self.turtle_real_x_pos_list.append(turtle[0])
        self.turtle_real_y_pos_list.append(turtle[1])
        # 6.4 - Draw clock
        font = pygame.font.Font(None, 24)

        self.time_elapsed = int(floor(time.time() - self.start_time))
        if self.time_dependend:
            survivedtext = font.render(
                str(self.TIME - self.time_elapsed), True, (0, 0, 0))
        else:
            survivedtext = font.render(
                str(self.time_elapsed), True, (0, 0, 0))

        self.screen.blit(survivedtext, (self.width / 2, 10))

        game_num_text = font.render("Game: " + str(self.experiment) + "/" + str(MAX_STEPS), True, (0, 0, 0))

        self.screen.blit(game_num_text, (90, 10))

        try:
            self.button("reset", 2, 2, 80, 40, BLUE, bright_green, "reset")
            self.button("pause", 2, 42, 80, 40, WHITE, bright_green, "pause")
        except pygame.error:
            print("An exception occurred")


        # 7 - update the screen
        pygame.display.flip()
        # 8 - loop through the events
        for event in pygame.event.get():
            # check if the event is the X button
            if event.type == pygame.QUIT:
                # if it is quit the game
                pygame.quit()
                exit(0)

        self.accel_x = x_data * accel_rate_x
        self.accel_y = y_data * accel_rate_y
        
        # print("Action: %f.\n Accel: %f.\nVel: %f." % (data[1], self.accel_y, self.vel_y))
        self.vel_x += self.accel_x
        self.vel_y += self.accel_y

        current_pos_x, current_pos_y = self.turtle_pos
        next_pos_x = self.turtle_pos[0] + self.vel_x
        next_pos_y = self.turtle_pos[1] - self.vel_y

        # check collision on the x axis
        if 0 < next_pos_x < self.width - 64:
            if (self.height / 2) - 64 < current_pos_y < (self.height / 2) \
                    and (next_pos_x < self.limit2 - 34 or next_pos_x + 34 > self.limit1):
                self.vel_x = 0
        else:
            self.vel_x = 0
        self.turtle_pos[0] += self.vel_x

        if 0 < next_pos_y < self.height - 64:
            if 0 < self.turtle_pos[0] < self.limit2 - 34 or self.limit1 - 34 < self.turtle_pos[0] < self.width:
                if self.height / 2 - 64 <= next_pos_y <= self.height / 2:
                    self.vel_y = 0
        else:
            self.vel_y = 0
        self.turtle_pos[1] -= self.vel_y

        self.accel_x_list.append(self.accel_x)
        self.accel_y_list.append(self.accel_y)

        self.vel_x_list.append(self.vel_x)
        self.vel_y_list.append(self.vel_y)

        self.time.append((time.time() - self.global_start_time) * 1e3)

        # print([self.vel_x, self.vel_y])
        # 10 - Win/Lose check
        if self.time_dependend and self.time_elapsed >= self.TIME:
            self.running = 0
            self.exitcode = 1
            self.timedOut = True

        if self.width - 40 > self.turtle_pos[0] > self.width - (80 + 40) \
                and 20 < self.turtle_pos[1] < (80 + 60 / 2 - 32):
            self.running = 0
            self.exitcode = 1
            self.finished = True  # This means final state achieved

        return time.time() - start_time


    def paused(self):
        largeText = pygame.font.SysFont("comicsansms",115)
        TextSurf, TextRect = text_objects("Paused", largeText)
        TextRect.center = ((self.width/2),(self.height/2))
        self.screen.blit(TextSurf, TextRect)
        

        while self.pause:
            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
            
            self.screen.fill(WHITE)

            font = pygame.font.Font(None, 64)
            text = font.render("Paused", True, BLACK)
            textRect = text.get_rect()
            textRect.centerx = self.screen.get_rect().centerx
            textRect.centery = self.screen.get_rect().centery + 24
            # screen.blit(youwin, (100, 100))
            self.screen.blit(text, (300, 300))

            self.button("Continue",150,450,100,50,GREEN,bright_green,"unpause")
            self.button("Quit",550,450,100,50,RED,bright_red,"quit")

            pygame.display.update()
            self.clock.tick(15)

    def waitScreen(self):
        pygame.font.init()
        font = pygame.font.Font(None, 64)
        text = font.render("Training... Please Wait.", True, RED)
        textRect = text.get_rect()
        textRect.centerx = self.screen.get_rect().centerx
        textRect.centery = self.screen.get_rect().centery + 24
        # screen.blit(youwin, (100, 100))
        self.screen.blit(text, (180, 300))
        pygame.display.flip()
        time.sleep(1)

    def getReward(self):
        if self.finished:
            return 100
        # elif not (700 - 64 < self.turtle_pos[0] + self.turtle_pos[1] < 900 - 64):
        #     return -10
        else:
            return -1

    def getState(self):
        # [x position. y position, detected wrist x position]
        # Size of observation space is 2
        return [self.accel_x, self.accel_y, self.turtle_pos[0], self.turtle_pos[1], self.vel_x, self.vel_y]

    def endGame(self):
        if self.exitcode == 1:
            pygame.font.init()
            font = pygame.font.Font(None, 64)
            text = font.render("Finished!", True, (0, 255, 0))
            textRect = text.get_rect()
            textRect.centerx = self.screen.get_rect().centerx
            textRect.centery = self.screen.get_rect().centery + 24
            # screen.blit(youwin, (100, 100))
            self.screen.blit(text, (250, 300))

        pygame.display.flip()
        time.sleep(1)
        pygame.display.quit()
        pygame.quit()

        # self.plot(self.time, self.accel_x_list, "accelaration_x_axis", 'accelaration x-axis','Running time since Game started(msec)', save=True)
        # self.plot(self.time, self.accel_y_list, "accelaration_y_axis", 'accelaration x-axis','Running time since Game started(msec)', save=True)
        # self.plot(self.time, self.vel_x_list, "velocity_x_axis", 'accelaration x-axis','Running time since Game started(msec)', save=True)
        # self.plot(self.time, self.vel_y_list, "velocity_y_axis", 'accelaration x-axis','Running time since Game started(msec)', save=True)
        # self.turtle_real_y_pos_list.reverse()
        # self.plot(self.turtle_real_x_pos_list, self.turtle_real_y_pos_list, "turtle_pos", 'y-position','x-position')

    # def game_loop(self):
    #     while self.running:
    #         self.play()
    #     self.endGame()

    # def plot(self, time_elpsd, list, figure_title, y_axis_name, x_axis_name, save=True):       
    #     plt.figure(figure_title)
    #     plt.grid()
    #     # plt.xticks(np.arange(0, time_elpsd[-1], step=500))
    #     # plt.yticks(np.arange(min(list), max(list), step=0.01))
    #     plt.plot(time_elpsd, list)
    #     plt.ylabel(y_axis_name)
    #     plt.xlabel(x_axis_name)
    #     if save:
    #         plt.savefig("/home/liger/catkin_ws/src/hand_direction/" + figure_title)
    #     else:
    #         plt.show()

