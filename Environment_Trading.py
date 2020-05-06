from enum import Enum

import gym
import torch
import numpy as np
import torchvision.transforms as T

from ImportData import get_data

# We will divide the whole history in windows of fixed length.
# Each window is an episode.
# Length of the window in days.
WINDOW_LENGTH = 30
# Amount to spend
SPEND_AMOUNT = 100


class Actions(Enum):
    WAIT = 1
    BUY = 2
    SELL = 3


class TradingEnvironment():
    def __init__(self):
        self.open_history, self.hours_history = get_data()

        # The current hour we are in, is the last element of the window
        self.current_window_end = WINDOW_LENGTH

        self.current_bought = 0

        self.has_window = True

        self.reset()

    # Moves the window so the last element is at the next 0 hour, and sets the state as waiting
    def reset(self):
        self.current_bought = 0
        found_window = False
        for i in range(self.current_window_end, len(self.hours_history)):
            if self.hours_history[i] == 0:
                found_window = True
                self.current_window_end = i
                break
        if not found_window:
            self.has_window = False

    def take_action(self, action):
        if action == Actions.WAIT:
            # If at the end of the day and have bought, can't wait, have to sell
            if self.hours_history[self.current_window_end] == 23 and self.current_bought != 0:
                raise ValueError("Have to sell what you bought at the end of the day")
            # Move one hour forward
            self.current_window_end += 1
            # Give a reward of 0 for waiting
            reward = 0
            # End the episode if at the end of the day
            done = self.hours_history[self.current_window_end] == 0
            return reward, done

        if action == Actions.BUY:
            if self.current_bought != 0:
                raise ValueError("Can't buy, already bought")
            # If at the end of the day, can't buy
            if self.hours_history[self.current_window_end] == 23:
                raise ValueError("Can't buy at the end of the day")
            # Calculate how much we bought
            self.current_bought = SPEND_AMOUNT / self.open_history[self.current_window_end]
            self.current_window_end += 1
            reward = - SPEND_AMOUNT
            done = False
            return reward, done

        if action == Actions.SELL:
            if self.current_bought == 0:
                raise ValueError("Trying to sell without buying first")
            reward = self.current_bought * self.open_history[self.current_window_end]
            self.current_window_end += 1
            self.current_bought = 0
            # End episode if sell
            done = True
            return reward, done

    def get_state(self):
        return self.current_bought, self.open_history[self.current_window_end - WINDOW_LENGTH:self.current_window_end]

    def get_possible_actions(self):
        if self.current_bought == 0:
            if self.hours_history[self.current_window_end] == 23:
                return [Actions.WAIT]
            else:
                return [Actions.BUY, Actions.WAIT]
        else:
            if self.hours_history[self.current_window_end] == 23:
                return [Actions.SELL]
            else:
                return [Actions.WAIT, Actions.SELL]

    def has_episodes(self):
        return self.has_window
