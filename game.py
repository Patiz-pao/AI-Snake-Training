import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import csv
import pandas as pd
from openpyxl import load_workbook
import json

pygame.init()

font = pygame.font.Font('arial.ttf', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# สีต่างๆ
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 176, 12)

BLOCK_SIZE = 20
RATE = 50

w = 720
h = 640

def load_bsc():
    try:
        with open('value/data.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                return int(row[0])
    except FileNotFoundError:
        return 0

class GameAI:
    def __init__(self):
        self.w = w
        self.h = h
        self.bsc = load_bsc()
        self.steps = []
        self.food1 = None
        self.food2 = None
        self.food3 = None
        self.food4 = None
        self.food5 = None

        self.last_button_press_time = 0
        self.button_cooldown = 150

        # โหลดภาพ
        self.head_right_img = pygame.transform.scale(pygame.image.load("image/head_right.png"), (BLOCK_SIZE, BLOCK_SIZE))
        self.head_left_img = pygame.transform.scale(pygame.image.load("image/head_left.png"), (BLOCK_SIZE, BLOCK_SIZE))
        self.head_up_img = pygame.transform.scale(pygame.image.load("image/head_up.png"), (BLOCK_SIZE, BLOCK_SIZE))
        self.head_down_img = pygame.transform.scale(pygame.image.load("image/head_down.png"), (BLOCK_SIZE, BLOCK_SIZE))

        self.food1_img = pygame.transform.scale(pygame.image.load("image/x-png-22.png"), (BLOCK_SIZE, BLOCK_SIZE))
        self.food2_img = pygame.transform.scale(pygame.image.load("image/x-png-22.png"), (BLOCK_SIZE, BLOCK_SIZE))
        self.food3_img = pygame.transform.scale(pygame.image.load("image/x-png-22.png"), (BLOCK_SIZE, BLOCK_SIZE))
        self.food4_img = pygame.transform.scale(pygame.image.load("image/x-png-22.png"), (BLOCK_SIZE, BLOCK_SIZE))
        self.food5_img = pygame.transform.scale(pygame.image.load("image/x-png-22.png"), (BLOCK_SIZE, BLOCK_SIZE))

        self.body_right_img = pygame.transform.scale(pygame.image.load("image/body_right.png"), (BLOCK_SIZE, BLOCK_SIZE))
        self.body_left_img = pygame.transform.scale(pygame.image.load("image/body_left.png"), (BLOCK_SIZE, BLOCK_SIZE))
        self.body_up_img = pygame.transform.scale(pygame.image.load("image/body_up.png"), (BLOCK_SIZE, BLOCK_SIZE))
        self.body_down_img = pygame.transform.scale(pygame.image.load("image/body_down.png"), (BLOCK_SIZE, BLOCK_SIZE))

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Game')
        self.clock = pygame.time.Clock()
        self.reset()

    def change_game_speed(self, keys):
        current_time = pygame.time.get_ticks()
        global RATE
        if current_time - self.last_button_press_time >= self.button_cooldown:
            if keys[pygame.K_x]:
                RATE += 10
                self.last_button_press_time = current_time
            elif keys[pygame.K_z]:
                RATE -= 10
                self.last_button_press_time = current_time

            if RATE < 10 :
                RATE = 10
            elif RATE > 100 :
                RATE = 100

    def handle_key_events(self):
        keys = pygame.key.get_pressed()
        self.change_game_speed(keys)

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.food1 = None
        self.food2 = None
        self.food3 = None
        self.food4 = None
        self._place_food()
        self.frame_iteration = 0

    def save_steps(self):
        df = pd.DataFrame(self.steps)
        df.to_excel('step/snake_steps.xlsx', index=False)
        wb = load_workbook('step/snake_steps.xlsx')
        wb.save('step/snake_steps.xlsx')


    def _place_food(self):
        for _ in range(5):
            while True:
                x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
                y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

                food_position = Point(x, y)

                if food_position not in self.snake and food_position != self.food and food_position != self.food1 and food_position != self.food2 and food_position != self.food3 and food_position != self.food4 and food_position != self.food5:
                    if _ == 0:
                        self.food = food_position
                    elif _ == 1:
                        self.food1 = food_position
                    elif _ == 2:
                        self.food2 = food_position
                    elif _ == 3:
                        self.food3 = food_position
                    elif _ == 4:
                        self.food4 = food_position
                    break



    def play_step(self, action):   
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # เคลื่อนไหวงูและอัปเดต UI
        self._move(action)
        self.snake.insert(0, self.head)

        self.steps.append({
            'Frame': self.frame_iteration,
            'Head_x': self.head.x,
            'Head_y': self.head.y,
            'Direction': self.direction.name,
            'Food_x': self.food.x,
            'Food_y': self.food.y,
            'Score': self.score
        })

        reward = 0
        bsc = 0
        game_over = False

        if self.head == self.food:
            self.score += 1
            bsc = load_bsc()
            if self.score >= self.bsc:
                self.bsc = self.score

            if not bsc or self.score > bsc:
                with open('value/data.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.score])
            reward = 10
            self._place_food()

        if self.head == self.food1:
            self.score -= 1
            self.food1 = None
            reward = -10
            self._place_food()
        elif self.head == self.food2:
            self.score -= 1
            self.food2 = None
            reward = -10
            self._place_food()
        elif self.head == self.food3:
            self.score -= 1
            self.food3 = None
            reward = -10
            self._place_food()
        elif self.head == self.food4:
            self.score -= 1
            self.food4 = None
            reward = -10
            self._place_food()
        elif self.head == self.food5:
            self.score -= 1
            self.food5 = None
            reward = -10
            self._place_food()

        else:
            self.snake.pop()

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            if self.score > self.bsc:
                self.bsc = self.score
                with open('value/data.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.score])
            return reward, game_over, self.score, bsc
        
        self._update_ui()
        self.clock.tick(RATE)
        return reward, game_over, self.score, self.bsc
    
    def get_formatted_steps(self):
        return json.dumps(self.steps, indent=4, sort_keys=True)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        grass_image = pygame.image.load("image/background.jpg")
        grass_image = pygame.transform.scale(grass_image, (self.w, self.h))
        self.display.blit(grass_image, (0, 0))

        for i, pt in enumerate(self.snake):
            if i == 0:
                if self.direction == Direction.RIGHT:
                    head_img = self.head_right_img
                elif self.direction == Direction.LEFT:
                    head_img = self.head_left_img
                elif self.direction == Direction.UP:
                    head_img = self.head_up_img
                elif self.direction == Direction.DOWN:
                    head_img = self.head_down_img
                self.display.blit(head_img, (pt.x, pt.y))
            else:
                BLUE3 = (39, 117, 163)
                pygame.draw.rect(self.display, BLUE3, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLACK, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), 1)
                if i == len(self.snake) - 1:
                    pygame.draw.rect(self.display, BLUE3, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                    pygame.draw.rect(self.display, BLACK, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), 1)
                else:
                    pygame.draw.rect(self.display, BLACK, (pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE), 1)

        food_img = pygame.image.load("image/star.png") if (self.frame_iteration // 10) % 2 == 0 else pygame.image.load("image/star2.png")
        food_img = pygame.transform.scale(food_img, (BLOCK_SIZE, BLOCK_SIZE))
        self.display.blit(food_img, (self.food.x, self.food.y))

        if self.food1:
            self.display.blit(self.food1_img, (self.food1.x, self.food1.y))

        if self.food2:
            self.display.blit(self.food2_img, (self.food2.x, self.food2.y))
        
        if self.food3:
            self.display.blit(self.food3_img, (self.food3.x, self.food3.y))

        if self.food4:
            self.display.blit(self.food4_img, (self.food4.x, self.food4.y))

        if self.food5:
            self.display.blit(self.food5_img, (self.food5.x, self.food5.y))

        text = font.render("Score: " + str(self.score), True, BLACK)
        text_rect = text.get_rect()
        pygame.draw.rect(text, BLACK, text_rect, 1)

        text_bsc = font.render("Best Score: " + str(self.bsc), True, BLACK)
        text_bsc_rect = text_bsc.get_rect()
        pygame.draw.rect(text_bsc, BLACK, text_bsc_rect, 1)

        speed_text = font.render("Speed: " + str(RATE), True, BLACK)
        speed_text_rect = speed_text.get_rect()
        speed_text_rect.bottomright = (self.w, self.h)
        self.display.blit(speed_text, speed_text_rect)


        self.display.blit(text, [0, 0])
        self.display.blit(text_bsc, [self.w - text_bsc.get_width(), 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)
