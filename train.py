import torch
import random
import numpy as np
import csv
from collections import deque
from game import GameAI, Direction, Point
from model import LQN, Train_Q, input_size, hidden_size, output_size
from plot import plot
import pygame
import json

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN = (0,176,12)

font = pygame.font.Font('arial.ttf', 25)

class Agent:

    def __init__(self):
        self.r_game = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = LQN(input_size, hidden_size, output_size)
        self.trainer = Train_Q(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):

        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            dir_l,dir_r,dir_u,dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
            ]

        return np.array(state, dtype=int)

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
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 1 - self.r_game
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def load_bsc():
    try:
        with open('value/data.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                return int(row[0])
    except FileNotFoundError:
        return 0 
    

def train():


    plot_scores = []
    plot_mean_scores = []

    total_score = 0
    record = 0

    agent = Agent()
    game = GameAI()

    agent.model.load_state_dict(torch.load('model/model.pth'))
    
    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score, bsc = game.play_step(final_move)

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        game.handle_key_events()

        if done:
            game.reset()
            agent.r_game += 1
            agent.train_long_memory()
            bsc = load_bsc()

            if score > record:
                record = score
                agent.model.save()
                game.save_steps()
            formatted_steps = game.get_formatted_steps()

            print(f'\033[92mGame {agent.r_game}\033[0m',f'\033[93mRecord : {record}\033[0m')
            print(formatted_steps)
            print('\n')
            print(f'\033[93m************************************************        Sum Score : {score}         ************************************************\033[0m')
            print(f'\033[92m************************************************        Best Score : {bsc}       ************************************************\033[0m')
            print('\033[91m************************************************          Game Over           ************************************************\033[0m')

            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.r_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def main():
    agent = Agent()
    game = GameAI()

    agent.model.load_state_dict(torch.load('model/model.pth'))
    bsc = load_bsc()

    # เริ่มเล่นเกมส์
    while True:
        game.handle_key_events()
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score, _ = game.play_step(final_move)
        state_new = agent.get_state(game)

        if done:
            game.reset()
            print('Game', agent.r_game, 'Score', score)
            print('Best Score:', bsc)

if __name__ == '__main__':
    main()