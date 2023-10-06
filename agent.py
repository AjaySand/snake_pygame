# agent.py

import numpy as np
import random
import torch
from collections import deque
from snake import Game
import time

ACTIONS = {"L": (-1, 0), "R": (1, 0), "D": (0, -1), "U": (0, 1)}
ACTIONS_LIST = ["L", "R", "D", "U"]

MAX_MEMORY = 100_000
BATCH_SIZE = 10
LR = 0.0001
RANDOMNESS = 0.1


class Agent(object):
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)

        # model: model, trainer

    def train(self):
        state = self.state_history[-1][0]

    def random_action(self, game):
        # make a copy of all the actions
        actions = ACTIONS_LIST.copy()

        dir_l = game.snake.head.pos == ACTIONS.get('L')
        dir_r = game.snake.head.pos == ACTIONS.get('R')
        dir_u = game.snake.head.pos == ACTIONS.get('U')
        dir_d = game.snake.head.pos == ACTIONS.get('D')

        # discard the direction opposite that the snake is currently moving
        if dir_l:
            actions.remove('R')
        elif dir_r:
            actions.remove('L')
        elif dir_u:
            actions.remove('D')
        elif dir_d:
            actions.remove('U')

        return ACTIONS[random.choice(actions)]

    def get_state(self, game):
        direction = [game.snake.dir_x, game.snake.dir_y]

        # extract gird
        grid = np.zeros((20, 20))
        for part in game.snake.body:
            # Ensure that the snake's position is within bounds before updating grid
            if 0 <= part.pos[0] < 20 and 0 <= part.pos[1] < 20:
                grid[part.pos[0], part.pos[1]] = -1

        # extract food
        grid[game.snack.pos] = 1

        # grid = grid.flatten()

        return np.append(direction, grid)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # self.trainer.train_step(state, action, reward, next_state, done)
        pass

    def get_action(self, state, game):
        self.epsilon = 100 - self.n_games
        next_action = self.random_action(game)

        if random.random() > self.epsilon:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            next_action = ACTIONS_LIST[torch.argmax(prediction).item()]

        return next_action



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = Game()

    while True:
        current_state = agent.get_state(game)

        # predict next move
        action = agent.get_action(current_state, game)


        # perform move and get new state
        reward, done, score = game.step(action)
        # print('action: ', action, 'pos', game.snake.head.pos)

        try:
            new_state = agent.get_state(game)
        except IndexError:
            print("IndexError")

        # train short memory
        # agent.train_short_memory(current_state, action, reward, new_state, done)

        # remember
        agent.remember(current_state, action, reward, new_state, done)

        if done:
            agent.n_games += 1
            # agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record", record)


if __name__ == "__main__":
    train()
