#!/usr/bin/env python
from __future__ import print_function

import time
import timeit
from math import floor

import pygame
from pygame.locals import *
from hand_direction.msg import action_agent

move_rate_x = 1*1e-3
move_rate_y = 1*1e-3


backgroundColor = (255, 255, 255)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


class Game:
    def __init__(self):
        self.TIME = 60
        self.start_time = time.time()
        self.time_elapsed = 0
        # 2 - Initialize the game
        self.reset_time = 2  # seconds
        pygame.init()
        self.width, self.height = 800, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.keys = [False, False, False, False]
        self.playerpos = [0, self.height - 64]
        self.reward = 0
        # 3 - Load images
        self.player = pygame.image.load("/home/liger/PycharmProjects/CollaborativeRL/resources/images/turtle.png")
        self.youwin = pygame.image.load("/home/liger/PycharmProjects/CollaborativeRL/resources/images/youwin.png")
        obst_x, obsty = self.width / 2, self.height / 2
        self.running = 1
        self.exitcode = 0
        # obstacle dimensions and position
        self.thing_startx = self.width / 4
        self.thing_starty = self.height / 2
        self.thing_width = 400
        self.thing_height = 30
        self.player_width = 64
        self.rotation = 45

        self.shift_y = self.shift_x = 0

        self.point_1a = (60 - 50, self.height - 60 - 50)
        self.point_2a = (self.width - 60 - 50, 60 - 50)
        self.point_1b = (60 + 50, self.height - 60 + 50)
        self.point_2b = (self.width - 60 + 50, 60 + 50)

        self.timedOut = self.finished = False

        self.limit1 = self.width / 2
        self.limit2 = self.width / 2 - 100
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

    def checkCollision(self):
        if self.playerpos[1] + 1 < self.thing_starty + self.thing_height:
            print('y crossover')

            if self.playerpos[0] > self.thing_startx and self.playerpos[1] < self.thing_startx + self.thing_width or \
                    self.playerpos[1] + self.player_width > self.thing_startx and self.playerpos[0] \
                    + self.player_width < self.thing_startx + self.thing_width:
                print('x crossover')
                return False
        return True


    def play(self, data=None):
        if data is None:
            data = [0, 0]
        x_data = int(data[0])
        y_data = int(data[1])
        # 5 - clear the screen before drawing it again
        self.screen.fill(backgroundColor)
        # 6 - draw the screen elements
        pygame.draw.circle(self.screen, RED, (self.width - 80, 80), 60)  # up-right

        self.obstacle3()
        self.screen.blit(self.player, self.playerpos)
        # 6.4 - Draw clock
        font = pygame.font.Font(None, 24)
        self.time_elapsed = int(floor(time.time() - self.start_time))
        survivedtext = font.render(
            str(self.TIME - self.time_elapsed), True, (0, 0, 0))

        self.screen.blit(survivedtext, (self.width / 2, 10))

        # 7 - update the screen
        pygame.display.flip()
        # 8 - loop through the events
        for event in pygame.event.get():
            # check if the event is the X button
            if event.type == pygame.QUIT:
                # if it is quit the game
                pygame.quit()
                exit(0)

        self.shift_y += y_data * move_rate_y
        self.shift_x += x_data * move_rate_x

        current_pos_x, current_pos_y = self.playerpos
        next_pos_x = self.playerpos[0] + self.shift_x
        next_pos_y = self.playerpos[1] - self.shift_y

        # check collision on the x axis
        if 0 < next_pos_x < self.width - 64:
            if (self.height / 2) - 64 < current_pos_y < (self.height / 2) \
                    and (next_pos_x < self.limit2 - 34 or next_pos_x + 34 > self.limit1):
                self.shift_x = 0
        else:
            self.shift_x = 0
        self.playerpos[0] += self.shift_x

        if 0 < next_pos_y < self.height - 64:
            if 0 < self.playerpos[0] < self.limit2 - 34 or self.limit1-34 < self.playerpos[0] < self.width:
                if self.height / 2 - 64 <= next_pos_y <= self.height / 2:
                    self.shift_y = 0
        else:
            self.shift_y = 0
        self.playerpos[1] -= self.shift_y


        # print([self.shift_x, self.shift_y])
        # 10 - Win/Lose check
        if self.time_elapsed >= self.TIME:
            self.running = 0
            self.exitcode = 1
            self.timedOut = True

        if self.width > self.playerpos[0] > self.width - (80 + 40) \
                and self.playerpos[1] < (80 + 60 / 2 - 32):
            self.running = 0
            self.exitcode = 1
            self.finished = True  # This means final state achieved

    def getReward(self):
        if self.timedOut:
            return -50
        elif self.finished:
            return 100
        # elif not (700 - 64 < self.playerpos[0] + self.playerpos[1] < 900 - 64):
        #     return -10
        else:
            return -1

    def getObservations(self):
        # [x position. y position, detected wrist x position]
        # Size of observation space is 2
        return [self.playerpos[0], self.playerpos[1]]

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
        # time.sleep(self.reset_time)

        # while 1:
        #     for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             pygame.quit()
        #     pygame.display.flip()

    def reset(self):
        self.__init__()


