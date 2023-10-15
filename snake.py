# snake.py

import math
import random
import pygame

# constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
BACKGROUND_COLOR = (0, 0, 0)  # White
ROWS = 20


class Cube(object):
    rows = ROWS
    w = SCREEN_WIDTH

    def __init__(self, start, dir_x=1, dir_y=0, color=(255, 0, 0)):
        self.pos = start
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.color = color

    def move(self, dir_x, dir_y):
        self.dir_x = dir_x
        self.dir_y = dir_y

        self.pos = (self.pos[0] + self.dir_x, self.pos[1] + self.dir_y)

    def draw(self, surface, eyes=False):
        dis = self.w // self.rows
        i = self.pos[0]  # row
        j = self.pos[1]  # column

        pygame.draw.rect(
            surface, self.color, (i * dis + 1, j * dis + 1, dis - 2, dis - 2)
        )

        if eyes:
            centre = dis // 2
            radius = 3
            circle_middle = (i * dis + centre - radius, j * dis + 8)
            circle_middle2 = (i * dis + dis - radius * 2, j * dis + 8)
            pygame.draw.circle(surface, (0, 0, 0), circle_middle, radius)
            pygame.draw.circle(surface, (0, 0, 0), circle_middle2, radius)


class Snake(object):
    body = []
    turns = {}

    def __init__(self, color, pos):
        self.color = color
        self.head = Cube(pos)
        self.body.append(self.head)
        self.dir_x = 0
        self.dir_y = 1

        self.add_cube()
        self.add_cube()
        self.add_cube()
        self.add_cube()

    def move(self, dir=None):
        ## add pressed keys to the turns dictionary
        new_dir_x = self.dir_x
        new_dir_y = self.dir_y

        keys = pygame.key.get_pressed()

        if dir:
            new_dir_x = dir[0]
            new_dir_y = dir[1]
        else:
            for event in pygame.event.get():
                for key in keys:
                    if keys[pygame.K_LEFT]:
                        new_dir_x = -1
                        new_dir_y = 0
                    elif keys[pygame.K_RIGHT]:
                        new_dir_x = 1
                        new_dir_y = 0
                    elif keys[pygame.K_UP]:
                        new_dir_x = 0
                        new_dir_y = -1
                    elif keys[pygame.K_DOWN]:
                        new_dir_x = 0
                        new_dir_y = 1

        # check if dir changed
        if (new_dir_x, new_dir_y) != (self.dir_x, self.dir_y):
            self.dir_x = new_dir_x
            self.dir_y = new_dir_y
            self.turns[self.head.pos[:]] = [self.dir_x, self.dir_y]

        ## Move the snake
        for i, c in enumerate(self.body):
            p = c.pos[:]

            # if the position is in the turns dictionary
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])

                # pop the last turn
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dir_x, c.dir_y)

    def wrap(self, dir=None):
        for i, c in enumerate(self.body):
            p = c.pos[:]

            if c.dir_x == -1 and c.pos[0] < 0:
                c.pos = (c.rows - 1, c.pos[1])
            elif c.dir_x == 1 and c.pos[0] > (c.rows - 1):
                c.pos = (0, c.pos[1])

            elif c.dir_y == 1 and c.pos[1] > (c.rows - 1):
                c.pos = (c.pos[0], 0)
            elif c.dir_y == -1 and c.pos[1] < 0:
                c.pos = (c.pos[0], c.rows - 1)

    def reset(self, pos):
        self.head = Cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dir_x = 0
        self.dir_y = 1

        self.add_cube()
        self.add_cube()
        self.add_cube()
        self.add_cube()

    def add_cube(self):
        tail = self.body[-1]
        dx, dy = tail.dir_x, tail.dir_y

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), dx, dy))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), dx, dy))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), dx, dy))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), dx, dy))

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Snake Game")

        pygame.init()
        pygame.font.init()

        self.clock = pygame.time.Clock()
        self.clock.tick(120)
        self.snacks = []
        self.num_snacks = 1

        self.snake = Snake((255, 0, 0), (10, 10))

        for i in range(self.num_snacks):
            self.snacks.append(Cube(self.random_snack(ROWS, self.snake), color=(0, 255, 0)))

    def step(self, action=None, step=0):
        reward = 5
        is_done = 0

        # Update game logic here
        self.snake.move(action)

        # check for collision with snack
        for snack in self.snacks:
            if self.snake.body[0].pos == snack.pos:
                self.snake.add_cube()
                self.snacks.remove(snack)
                self.snacks.append(Cube(self.random_snack(ROWS, self.snake), color=(0, 255, 0)))

                reward = 25
                break

        self.snake.wrap(action)
        current_score = len(self.snake.body)

        # checking for collision with itself
        for x in range(len(self.snake.body)):
            if self.snake.body[x].pos in list(
                map(lambda z: z.pos, self.snake.body[x + 1 :])
            ):
                # print("Score: ", len(self.snake.body))
                self.reset()

                reward = -5000
                is_done = 1

                break

        # Draw game elements here and Update the display
        # if step % 100 == 0:
        self.redraw_window(self.screen)

        # Return reward, done, score
        return reward, is_done, current_score - 3

    def setup(self, external_controls=False):
        if not external_controls:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.running = True
            while self.running:
                pygame.time.delay(180)
                self.step()

    def draw_grid(self, rows, width, surface):
        size_between = width // rows
        x = 0
        y = 0

        for i in range(rows):
            x += size_between
            y += size_between

            # Draw horizontal lines
            pygame.draw.line(surface, (100, 100, 100), (x, 0), (x, width))

            # Draw vertical lines
            pygame.draw.line(surface, (100, 100, 100), (0, y), (width, y))

    def redraw_window(self, surface):
        surface.fill(BACKGROUND_COLOR)
        self.snake.draw(surface)

        for snack in self.snacks:
            snack.draw(surface)

        self.draw_grid(ROWS, SCREEN_WIDTH, surface)

        # draw text on screen
        font = pygame.font.SysFont("comicsans", 40)
        text = font.render("Score: " + str(len(self.snake.body)), True, (255, 255, 255))
        surface.blit(text, (SCREEN_WIDTH - 150, 10))

        pygame.display.flip()

    def random_snack(self, rows, snake):
        positions = self.snake.body

        while True:
            x = random.randrange(rows)
            y = random.randrange(rows)

            if len(list(filter(lambda z: z.pos == (x, y), positions))) > 0:
                continue
            elif len(list(filter(lambda z: z.pos == (x, y), self.snacks))) > 0:
                continue
            else:
                break

        return (x, y)

    def reset(self):
        print("\n")
        self.snake.reset((10, 10))
        self.snacks = []

        for i in range(self.num_snacks):
            self.snacks.append(Cube(self.random_snack(ROWS, self.snake), color=(0, 255, 0)))


def main():
    game = Game()
    game.setup(external_controls=False)

    for i in range(100):
        game.step(game.random_action())
        pygame.time.delay(1000)

    # wait for player to close window
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


if __name__ == "__main__":
    main()
