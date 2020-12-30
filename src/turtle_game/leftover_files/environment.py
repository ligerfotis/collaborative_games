#!/mnt/34C28480C28447D6/PycharmProjects/maze3d_collaborative/venv/bin/python
from __future__ import print_function

import time
import timeit
from math import floor
import time
import pygame
from pygame.locals import *
from collaborative_games.msg import action_agent
import matplotlib.pyplot as plt
import numpy as np
from hyperparams_ur10 import MAX_STEPS, TIME_PER_TURN, ACCEL_RATE
import rospkg
import random

accel_rate = ACCEL_RATE

dt=True

backgroundColor = (255, 255, 255)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
bright_red = (255, 0, 0)
bright_green = (0, 255, 0)

# path = "/home/fligerakis/catkin_ws/src/collaborative_games/src/"
rospack = rospkg.RosPack()
package_path = rospack.get_path("collaborative_games")

def text_objects(text, font):
    textSurface = font.render(text, True, BLACK)
    return textSurface, textSurface.get_rect()

def quit_game():
    pygame.display.quit()
    pygame.quit()
    exit(0)

class Game:
    def __init__(self):
        self.fps = 80

        self.experiment = 0
        self.TIME = TIME_PER_TURN
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
        # random starting position under the barriers`
        # self.turtle_pos = [random.randint(0, self.width-64), random.randint(700, self.height-64)]
        self.turtle_pos = [5, self.height - 64]
        
        self.reward = 0
        # 3 - Load images
        self.player = pygame.image.load(package_path + "/src/pictures/turtle.png").convert_alpha()
        self.youwin = pygame.image.load(package_path + "/src/pictures/youwin.png").convert_alpha()
        obst_x, obsty = self.width / 2, self.height / 2
        self.running = 1
        self.exitcode = 0
        # obstacle dimensions and position
        self.thing_startx = self.width / 4
        self.thing_starty = self.height / 2
        self.thing_width = 400
        self.thing_height = 30
        self.player_width = 64

        self.real_turtle_pos = None

        self.vel_y = self.vel_x = 0

        self.timedOut = False
        self.finished = False

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

        self.current_fps = None

        self.accel_x, self.accel_y, self.vel_x, self.vel_y = [0, 0, 0, 0]

        self.clock = pygame.time.Clock()
        self.time_dependend = True

        self.intro = True
        self.pause = False

        self.dt = None
        self.real_turtle_pos = self.turtle_pos
        pygame.display.update()


    def barriers_obstacle(self):
        pygame.draw.line(self.screen, BLACK, [0, self.height / 2], [self.limit2, self.height / 2])
        pygame.draw.line(self.screen, BLACK, [self.limit1, self.height / 2], [self.width, self.height / 2])

    def update_fps(self):
        self.dt = self.clock.tick(self.fps)
        fps_str = str(int(self.clock.get_fps()))
        self.current_fps = self.clock.get_fps()
        font = pygame.font.SysFont("Arial", 18)
        fps_text = font.render("FPS: " + fps_str, 1, pygame.Color("coral"))
        return fps_text


    def button(self, msg, x, y, w, h, ic, ac, action=None):
        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        # print(click)
        if x + w > mouse[0] > x and y + h > mouse[1] > y:
            pygame.draw.rect(self.screen, ac, (x, y, w, h))
            if click[0] == 1 and action is not None:
                if action == "play":
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

    def play(self, data=None, total_games=MAX_STEPS, control_mode="accel_dir"):
        """
        Executes one action (data) and renders the game.
        """

        # print(data)
        self.start_time = time.time()
        if data is None:
            data = [0, 0]

        x_data = data[0]
        y_data = data[1]

        # clear the screen before drawing it again
        self.screen.fill(backgroundColor)
        # draw the screen elements
        #pygame.draw.circle(self.screen, RED, (self.width - 80, 80), 60)  # up-right
        
        pygame.draw.rect(self.screen, RED, (self.width - 160, 40, 100, 100 ))
        
        self.barriers_obstacle()
        
        self.screen.blit(self.update_fps(), (self.width / 2 - 150, 7))
        # self.screen.blit(self.update_fps(), (self.width / 2 - 150, 7))

        # actions are the acceleration directions
        if control_mode == "accel_dir":
            self.accel_x = x_data * accel_rate
            self.accel_y = y_data * accel_rate
        
            # print("Action: %f.\n Accel: %f.\nVel: %f." % (data[1], self.accel_y, self.vel_y))
            self.vel_x += self.accel_x
            self.vel_y += self.accel_y

        # actions are the commanded velocities
        elif control_mode == "vel":
            self.vel_x = x_data
            self.vel_y = y_data 
        elif control_mode == "accel":
            self.accel_x = x_data * accel_rate
            self.accel_y = y_data * accel_rate
        
            # print("Action: %f.\n Accel: %f.\nVel: %f." % (data[1], self.accel_y, self.vel_y))
            self.vel_x += self.accel_x
            self.vel_y += self.accel_y
       

        current_pos_x = self.real_turtle_pos[0]
        current_pos_y = self.real_turtle_pos[1]

        next_pos_x = self.turtle_pos[0] + self.vel_x * self.dt
        next_pos_y = self.turtle_pos[1] - self.vel_y * self.dt

        # check collision on the x axis
        if 0 < next_pos_x < self.width - 64:
            #check collision with barriers
            if (self.height / 2) - 64 < current_pos_y < (self.height / 2) \
                    and (next_pos_x < self.limit2 - 34 or next_pos_x + 34 > self.limit1):
                self.vel_x = 0
        else:
            if next_pos_x <= 0:
                self.vel_x = 0
            else:
                self.vel_x = 0

        self.turtle_pos[0] += self.vel_x * self.dt

        # check collision on the y axis
        if 0 <= next_pos_y <= self.height - 64:
            #check collision with barriers
            if 0 <= self.turtle_pos[0] < self.limit2 - 34 or self.limit1 - 34 < self.turtle_pos[0] < self.width:
                if current_pos_y >= self.height / 2:
                    if next_pos_y <= self.height / 2:
                        # print("here1")
                        self.vel_y = 0
                elif current_pos_y + 64 <= self.height / 2:
                    if next_pos_y +64 >= self.height / 2:
                        # print("here2")
                        self.vel_y = 0
        else:
            if next_pos_y < 0:
                self.vel_y = 0
            else:
                self.vel_y = 0

        if self.vel_y > 1:
            self.vel_y = 1
        if dt:
            self.vel = self.vel_y * self.dt
        self.turtle_pos[1] -= self.vel_y 

        # print("Action: %f.\n Accel: %f.\nVel: %f.\nPos: %f." % (data[0], self.accel_x, self.vel_x, next_pos_x))


        self.real_turtle_pos = self.screen.blit(self.player, self.turtle_pos)

        self.turtle_real_x_pos_list.append(self.real_turtle_pos[0])
        self.turtle_real_y_pos_list.append(self.real_turtle_pos[1])
        # Draw clock
        font = pygame.font.Font(None, 24)
        self.time_elapsed = int(floor(time.time() - self.start_time))
        if self.time_dependend:
            survivedtext = font.render( "Time: " + 
                str(self.TIME - self.time_elapsed), True, (0, 0, 0))
        else:
            survivedtext = font.render( "Time: " + str(self.time_elapsed), True, (0, 0, 0))

        self.screen.blit(survivedtext, (self.width / 2, 10))

        game_num_text = font.render("Game: " + str(self.experiment) + "/" + str(total_games), True, (0, 0, 0))

        self.screen.blit(game_num_text, (90, 10))

        try:
            self.button("reset", 2, 2, 80, 40, BLUE, bright_green, "reset")
            self.button("pause", 2, 42, 80, 40, WHITE, bright_green, "pause")
        except pygame.error:
            print("An exception occurred")

        # update the screen
        pygame.display.flip()

        # loop through the events
        for event in pygame.event.get():
            # check if the event is the X button
            if event.type == pygame.QUIT:
                # if it is quit the game
                pygame.quit()
                exit(0)

        # print("Pos x: %f. Vel x: %f. Accel x: %f" % (self.turtle_pos[0], self.vel_x, self.accel_x))
        # print("Pos y: %f. Vel y: %f. Accel y: %f" % (self.turtle_pos[1], self.vel_y, self.accel_y))

        self.accel_x_list.append(self.accel_x)
        self.accel_y_list.append(self.accel_y)

        self.vel_x_list.append(self.vel_x)
        self.vel_y_list.append(self.vel_y)

        self.time.append((time.time() - self.global_start_time) * 1e3)
        # self.clock.tick()
        # return time.time() - start_time
        return time.time() - self.start_time


    def paused(self):
        pause_start_time = time.time()
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
        pause_duration = time.time() - pause_start_time
        self.start_time += pause_duration

    def waitScreen(self, msg1, msg2=None, duration=None):
        self.screen.fill(WHITE)
        pygame.font.init()
        font_text1 = pygame.font.Font(None, 40)
        font_text2 = pygame.font.Font(None, 40)
        font_text3 = pygame.font.Font(None, 50)

        text = font_text1.render(msg1, True, RED)
        textRect = text.get_rect()
        textRect.centerx = self.screen.get_rect().centerx
        textRect.centery = self.screen.get_rect().centery + 24
        # screen.blit(youwin, (100, 100))
        self.screen.blit(text, (180, 300))
        
        pygame.display.flip()
        
        if duration is not None:
            for step in range(duration):
                self.screen.fill(WHITE)
                self.screen.blit(text, (180, 300))

                text2 = font_text2.render("Game Starts in: ", True, BLACK)
                self.screen.blit(text2, (250, 350))

                font_num = pygame.font.Font(None, 50)
                time_text = font_num.render(str(duration - step - 1), True, BLACK)
                self.screen.blit(time_text, (480, 345))

                if msg2 is not None:
                    text3 = font_text3.render(msg2, True, BLUE)
                    self.screen.blit(text3, (170, 400))

                pygame.display.flip()
                time.sleep(1)

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


