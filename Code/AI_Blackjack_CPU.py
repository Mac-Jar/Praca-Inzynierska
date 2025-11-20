# engine.py
import random
from utils import print_policy

# wartości kart 2-10 + J/Q/K (10) + Ace (11)
CARD_VALUES = [2,3,4,5,6,7,8,9,10,10,10,10,11]

class BlackjackEngineCPU:
    """
    Minimalny engine, logika identyczna do Twojej klasy Blackjack (ocena asów, dealer hits <17).
    Nie używa str/obiektów kart.
    """

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def draw_card(self):
        return random.choice(CARD_VALUES)

    def evaluate_hand(self, cards):
        total = 0
        aces = 0
        for v in cards:
            total += v
            if v == 11:
                aces += 1
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total

    def new_game(self):
        player = [self.draw_card(), self.draw_card()]
        dealer = [self.draw_card(), self.draw_card()]
        return player, dealer

    def step_player(self, player_cards, action):
        # action: 1 hit, 0 stand
        if action == 1:
            player_cards.append(self.draw_card())
            if self.evaluate_hand(player_cards) > 21:
                return player_cards, True, True  # done, busted
            else:
                return player_cards, False, False
        return player_cards, True, False

    def step_dealer(self, dealer_cards):
        while self.evaluate_hand(dealer_cards) < 17:
            dealer_cards.append(self.draw_card())
        return dealer_cards

    def get_state(self, player_cards, dealer_cards):
        player_sum = self.evaluate_hand(player_cards)
        dealer_card = dealer_cards[0]
        if dealer_card > 10:
            dealer_card = 10
        usable_ace = 1 if (11 in player_cards and self.evaluate_hand(player_cards) <= 21) else 0
        if player_sum > 21:
            player_sum = 22
        return (player_sum, dealer_card, usable_ace)

    def play_full_episode(self, policy_fn):
        """
        policy_fn(state) -> action (0 or 1)
        Returns: visited_states (list of tuples), actions (list), reward (float)
        """
        player, dealer = self.new_game()
        visited_states = []
        actions = []

        # player turn
        while True:
            state = self.get_state(player, dealer)
            action = policy_fn(state)
            visited_states.append(state)
            actions.append(action)

            player, done, busted = self.step_player(player, action)
            if done:
                break
            if busted:
                return visited_states, actions, -1.0

        # dealer
        dealer = self.step_dealer(dealer)

        ps = self.evaluate_hand(player)
        ds = self.evaluate_hand(dealer)
        if ds > 21:
            reward = 1.0
        elif ps > ds:
            reward = 1.0
        elif ps == ds:
            reward = 0.0
        else:
            reward = -1.0

        return visited_states, actions, reward

# ai_cpu.py
import numpy as np
import random
# from engine import BlackjackEngineCPU

class AI_Blackjack_CPU:
    def __init__(self, epsilon=0.1):
        # Q[player_sum][dealer_card][usable_ace][action]
        self.Q = np.zeros((32, 11, 2, 2), dtype=np.float64)
        self.n = np.zeros((32, 11, 2, 2), dtype=np.int64)
        self.epsilon = epsilon
        self.engine = BlackjackEngineCPU()

    def choose_action(self, state):
        player_sum, dealer_card, usable_ace = state
        if random.random() < self.epsilon:
            return random.choice([0,1])
        arr = self.Q[player_sum, dealer_card, usable_ace]
        max_val = np.max(arr)
        choices = np.flatnonzero(arr == max_val)
        return int(np.random.choice(choices))

    def choose_action_final(self, state):
        player_sum, dealer_card, usable_ace = state
        return int(np.argmax(self.Q[player_sum, dealer_card, usable_ace]))

    def play_episode_and_update(self):
        visited_states, actions, reward = self.engine.play_full_episode(self.choose_action)
        # update Monte-Carlo average for each visited (s,a)
        for state, action in zip(visited_states, actions):
            ps, dc, ua = state
            # incremental average:
            cur_n = self.n[ps,dc,ua,action]
            self.Q[ps,dc,ua,action] = (self.Q[ps,dc,ua,action]*cur_n + reward) / (cur_n + 1)
            self.n[ps,dc,ua,action] += 1
        return reward

    def train(self, episodes=100000, report_every=10000):
        for i in range(episodes):
            self.play_episode_and_update()
            if (i+1) % report_every == 0:
                print(f"[CPU] Trained {i+1}/{episodes}")

    def evaluate_policy(self, print_n=False):
        print_policy(self.Q, self.n, title="CPU Policy", print_n=print_n)

