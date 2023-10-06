# agent.py

import numpy as np
import random
import torch
from collections import deque
from snake import Game, ROWS
import time
from model import Linear_QNet, QTrainer
from helper import plot

DIRECTIONS_LIST = {"L": (-1, 0), "R": (1, 0), "U": (0, -1), "D": (0, 1)}
DIRECTIONS = ["L", "R", "D", "U"]
ACTIONS = [0, 1, 2]  # 0 = left, 1 = right, 2 = straight
ACTIONS_LIST = {0: "L", 1: "R", 2: "S"}


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
RANDOMNESS = 0.1


class Agent(object):
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        # model: model, trainer
        self.model = Linear_QNet(ROWS * ROWS + 2, (ROWS * ROWS + 2) * 2, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def train(self):
        state = self.state_history[-1][0]

    # consider the snake's current direction
    # and randomly choose a direction that is not the opposite of the snake's current direction
    # return 'R', 'L' OR 'S'
    def random_action(self):
        return ACTIONS[random.randint(0, 2)]

    def translate_action_to_direction(self, action, game):
        direction = None

        dir = (game.snake.dir_x, game.snake.dir_y)

        dir_l = dir == DIRECTIONS_LIST.get("L")
        dir_r = dir == DIRECTIONS_LIST.get("R")
        dir_u = dir == DIRECTIONS_LIST.get("U")
        dir_d = dir == DIRECTIONS_LIST.get("D")

        # ACTIONS_LIST = {0: "L", 1: "R", 2: "S"}
        if dir_l:
            if action == 0:
                direction = "D"
            elif action == 1:
                direction = "U"
            elif action == 2:
                direction = "L"
        elif dir_r:
            if action == 0:
                direction = "U"
            elif action == 1:
                direction = "D"
            elif action == 2:
                direction = "R"
        elif dir_d:
            if action == 0:
                direction = "R"
            elif action == 1:
                direction = "L"
            elif action == 2:
                direction = "D"
        elif dir_u:
            if action == 0:
                direction = "L"
            elif action == 1:
                direction = "R"
            elif action == 2:
                direction = "U"

        return DIRECTIONS_LIST.get(direction)

    def get_state(self, game):
        direction = [game.snake.dir_x, game.snake.dir_y]

        # extract gird
        grid = np.zeros((20, 20))
        for part in game.snake.body:
            # Ensure that the snake's position is within bounds before updating grid
            if 0 <= part.pos[0] < 20 and 0 <= part.pos[1] < 20:
                grid[part.pos[0], part.pos[1]] = -1

        # extract food
        for snack in game.snacks:
            grid[snack.pos] = 1

        # grid = grid.flatten()

        return np.append(direction, grid)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        print("train_long_memory")
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game):
        self.epsilon = 15 - self.n_games
        next_action = self.random_action()  # return 'R', 'L' OR 'S'

        if random.randint(0, 200) > self.epsilon:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # should return a tensor of size 3
            next_action = torch.argmax(prediction).item()

        return next_action


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    memory_size = 0

    agent = Agent()
    game = Game()

    memory_size = len(agent.memory)

    # load model
    agent.model.load()

    while True:
        # time.sleep(0.025)

        # predict next move
        current_state = agent.get_state(game)
        action = agent.get_action(current_state, game)

        # perform move and get new state
        reward, done, score = game.step(
            agent.translate_action_to_direction(action, game)
        )

        new_state = agent.get_state(game)

        # train short memory
        agent.train_short_memory(current_state, action, reward, new_state, done)

        # remember
        agent.remember(current_state, action, reward, new_state, done)

        if done:
            agent.n_games += 1
            agent.train_long_memory()

            current_memory_size = len(agent.memory)
            memory_size = current_memory_size

            if score > record:
                record = score
                agent.model.save()

            if agent.n_games % 20 == 0:
                agent.model.save(file_name='model_' + str(agent.n_games) + '.pth')

            # number of games, score, record, reward, old memory size, new memory size
            print("Game", agent.n_games, "Score", score, "Record", record, "Reward", reward, "Memory Size", memory_size, current_memory_size)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
