import pygame
pygame.init()

win = pygame.display.set_mode((500, 500))
pygame.display.set_caption("ACCELERATE")

def main():

    k = True

    thita = 40
    x = 250
    y = 400

    while k:

        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    thita = 40
                if event.key == pygame.K_RIGHT:
                    thita = 40
            if event.type == pygame.QUIT:
                k = False

        if keys[pygame.K_LEFT]:
            x -= 4
            pygame.time.delay(thita)
            if thita > 12:
                thita -= 1
        if keys[pygame.K_RIGHT]:
            x += 4
            pygame.time.delay(thita)
            if thita > 11:
                thita -= 1

        pygame.draw.rect(win, (255, 0, 0), (x, y, 10, 10))
        pygame.display.update()
        win.fill((0, 0, 0))

main()