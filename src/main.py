import pygame
from pygame.locals import *

move_rate = 10


backgroundColor = (255, 255, 255)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)


class Game:
    def __init__(self):
        # 2 - Initialize the game
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

    def obstacle(self,):
        point_1a = (60 - 50, self.height-60 -50)
        point_2a = (self.width-60-50, 60-50)
        point_1b = (60 + 50, self.height - 60 + 50)
        point_2b = (self.width - 60 + 50, 60 + 50)
        pygame.draw.line(self.screen, BLACK, point_1a, point_2a)
        pygame.draw.line(self.screen, BLACK, point_1b, point_2b)


    def checkCollision(self):
        if self.playerpos[1]+1 < self.thing_starty + self.thing_height:
            print('y crossover')

            if self.playerpos[0] > self.thing_startx and self.playerpos[1] < self.thing_startx + self.thing_width or self.playerpos[1] + self.player_width > self.thing_startx and self.playerpos[0] + self.player_width < self.thing_startx+self.thing_width:
                print('x crossover')
                return False
        return True


    def play(self, x_data):
        # 4 - keep looping through
        # while self.running:
        # 5 - clear the screen before drawing it again
        self.screen.fill(backgroundColor)
        # 6 - draw the screen elements
        # pygame.draw.circle(self.screen, RED, (80, 80), 60)  # up-left
        pygame.draw.circle(self.screen, RED, (self.width - 80, 80), 60)  # up-right
        # pygame.draw.circle(self.screen, RED, (80, self.height - 80), 60)  # down-left
        # pygame.draw.circle(self.screen, RED, (self.width - 80, self.height - 80), 60)  # down-right

        self.obstacle()

        self.screen.blit(self.player, self.playerpos)
        # 6.4 - Draw clock
        font = pygame.font.Font(None, 24)
        survivedtext = font.render(
            str((30000 - pygame.time.get_ticks()) / 60000) + ":" + str(
                (30000 - pygame.time.get_ticks()) / 1000 % 60).zfill(2), True, (0, 0, 0))

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
            if event.type == pygame.KEYDOWN:
                if event.key == K_w:
                    self.keys[0] = True
                elif event.key == K_a:
                    self.keys[1] = True
                elif event.key == K_s:
                    self.keys[2] = True
                elif event.key == K_d:
                    self.keys[3] = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    self.keys[0] = False
                elif event.key == pygame.K_a:
                    self.keys[1] = False
                elif event.key == pygame.K_s:
                    self.keys[2] = False
                elif event.key == pygame.K_d:
                    self.keys[3] = False
        # 9 - Move player
        if self.keys[0]:
            if self.playerpos[1] - move_rate > 0:
                self.playerpos[1] -= move_rate
        elif self.keys[2]:
            if self.playerpos[1] + move_rate < self.height - 64:
                self.playerpos[1] += move_rate
        # if self.keys[1]:
        #     if self.playerpos[0] - move_rate > 0:
        #         self.playerpos[0] -= move_rate
        # elif self.keys[3]:
        #     if self.playerpos[0] + move_rate < self.width - 64:
        #         self.playerpos[0] += move_rate
        final_shift = x_data*2
        print final_shift
        if self.height - 64 > self.playerpos[0] + final_shift > 0:
            self.playerpos[0] += final_shift
        

        # 10 - Win/Lose check
        if pygame.time.get_ticks() >= 30000:
            self.running = 0
            self.exitcode = 1
        # # check if in top-left circle
        # if self.playerpos[0] < (80 - 20) and self.playerpos[1] < (80 - 20):
        #     self.running = 0
        #     self.exitcode = 1
        # check if in top-right circle
        if self.width > self.playerpos[0] > self.width - (80 + 40) \
                and self.playerpos[1] < (80 + 60 / 2 - 32):
            self.running = 0
            self.exitcode = 1
        # # check if in bottom-left circle
        # if self.playerpos[0] < (80 - 20) and self.playerpos[1] in range(self.height - (80 + 60), self.height):
        #     self.running = 0
        #     self.exitcode = 1
        # # check if in bottom-right circle
        # if self.playerpos[0] in range(self.width - (80 + 60), self.width) and self.playerpos[1] in range(
        #         self.height - (80 + 60 / 2), self.height):
        #     self.running = 0
        #     self.exitcode = 1

    def endGame(self):
        if self.exitcode == 1:
            pygame.font.init()
            font = pygame.font.Font(None, 64)
            text = font.render("Reward: " + str(100), True, (0, 255, 0))
            textRect = text.get_rect()
            textRect.centerx = self.screen.get_rect().centerx
            textRect.centery = self.screen.get_rect().centery + 24
            # screen.blit(youwin, (100, 100))
            self.screen.blit(text, (250, 300))

        while 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit(0)
            pygame.display.flip()


# game = Game()
# while game.running:
#     game.play()
# game.endGame()
