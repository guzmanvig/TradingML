from enum import Enum

import gym
import torch
import numpy as np
import torchvision.transforms as T

from ImportData import get_data

# We will divide the whole history in windows of fixed length.
# Each window is an episode.
# Each window is divided in hours
WINDOW_DAYS = 30
# Length of the window in hours
#WINDOW_LENGTH = WINDOW_DAYS * 24
WINDOW_LENGTH = 3
# Amount to spend
SPEND_AMOUNT = 100


class Actions(Enum):
    WAIT = 0
    BUY = 1
    SELL = 2


class TradingEnvironment():
    def __init__(self):
        #self.exchange_history, self.hours_history = get_data()
        self.exchange_history = [1, 3, 5, 7, 9, 2, 4, 6, 8]
        self.hours_history = [15, 23, 0, 15, 23, 0, 15, 23, 0]

        # The current hour we are in, is the last element of the window
        self.current_window_end = WINDOW_LENGTH - 1

        self.old_exchange = 0
        self.old_time = -1

        self.has_window = True

    # Moves the window so the last element is at the next 0 hour, and sets the state as waiting
    def reset(self):
        self.old_exchange = 0
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
            # TODO: chequear limites, y si se pasa, vender, dar reward negativo y terminar episodio
            # If at the end of the day and have bought, can't wait, have to sell
            if self.hours_history[self.current_window_end] == 23 and self.old_exchange != 0:
                raise ValueError("Have to sell what you bought at the end of the day")
            # Move one hour forward
            self.current_window_end += 1
            # Give a reward of 0 for waiting
            reward = 0
            # End the episode if at the end of the day
            done = self.hours_history[self.current_window_end] == 0
            return reward, done

        if action == Actions.BUY:
            if self.old_exchange != 0:
                raise ValueError("Can't buy, already bought")
            # If at the end of the day, can't buy
            if self.hours_history[self.current_window_end] == 23:
                raise ValueError("Can't buy at the end of the day")
            # Calculate how much we bought
            self.old_exchange = self.exchange_history[self.current_window_end]
            self.old_time = self.hours_history[self.current_window_end]
            self.current_window_end += 1
            reward = 0
            done = False
            return reward, done

        if action == Actions.SELL:
            if self.old_exchange == 0:
                raise ValueError("Trying to sell without buying first")
            reward = self.exchange_history[self.current_window_end] - self.old_exchange
            self.current_window_end += 1
            self.old_exchange = 0
            self.old_time = -1
            # End episode if sell
            # TODO: terminar el episodio?
            done = True
            return reward, done

    def get_state(self):
        state = 0 if self.old_exchange == 0 else 1
        return self.old_exchange, self.old_time, state, self.exchange_history[self.current_window_end - WINDOW_LENGTH + 1:self.current_window_end + 1], \
               self.hours_history[self.current_window_end - WINDOW_LENGTH + 1:self.current_window_end + 1]

    def get_possible_actions(self):
        if self.old_exchange == 0:
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
