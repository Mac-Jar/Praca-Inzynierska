import numpy as np
import random
from utils import print_policy
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

class AI_Blackjack_CPU_ES:
    """
    Monte Carlo Exploring Starts (ES)
    zgodnie z książką Sutton & Barto.
    """

    def __init__(self):
        # Q[player_sum][dealer_card][usable_ace][action]
        self.Q = np.zeros((32, 11, 2, 2), dtype=np.float64)
        self.n = np.zeros((32, 11, 2, 2), dtype=np.int64)
        self.engine = BlackjackEngineCPU()

    # -----------------------------
    #   Losowy stan startowy ES
    # -----------------------------
    def random_state(self):
        player_sum = random.randint(12, 21)
        dealer_card = random.randint(1, 10)
        usable_ace = random.choice([0, 1])

        # Odwzorowanie player_sum na realistyczne karty (na potrzeby symulatora)
        cards = []
        rest = player_sum

        # jeśli usable_ace==1, daj asa jako 11
        if usable_ace == 1:
            cards.append(11)
            rest -= 11

        # generujemy karty o wartości 2-10 aż złożą sumę
        while rest > 0:
            v = min(rest, random.randint(2, 10))
            cards.append(v)
            rest -= v

        dealer_cards = [dealer_card, self.engine.draw_card()]

        return cards, dealer_cards

    # -----------------------------
    #   Polityka greedy (bez epsilon)
    # -----------------------------
    def get_greedy_action(self, state):
        ps, dc, ua = state
        return int(np.argmax(self.Q[ps, dc, ua]))

    # -----------------------------
    #   Pełny epizod ES z wymuszonym startem
    # -----------------------------
    def play_episode_exploring_starts(self):
        # wygeneruj losowy stan startowy
        player, dealer = self.random_state()

        # losowa pierwsza akcja (exploring start)
        first_action = random.choice([0, 1])

        visited_states = []
        actions = []

        # --- Pierwsza wymuszona akcja ---
        state = self.engine.get_state(player, dealer)
        visited_states.append(state)
        actions.append(first_action)

        player, done, busted = self.engine.step_player(player, first_action)
        if busted:
            return visited_states, actions, -1.0
        if done:
            # player stoi od razu → przechodzi do dealera
            dealer = self.engine.step_dealer(dealer)
        else:
            # --- Pozostałe akcje według polityki greedy ---
            while True:
                state = self.engine.get_state(player, dealer)
                action = self.get_greedy_action(state)
                visited_states.append(state)
                actions.append(action)

                player, done, busted = self.engine.step_player(player, action)
                if busted:
                    return visited_states, actions, -1.0
                if done:
                    break

            dealer = self.engine.step_dealer(dealer)

        # --- Nagroda ---
        ps = self.engine.evaluate_hand(player)
        ds = self.engine.evaluate_hand(dealer)
        if ds > 21:
            reward = 1.0
        elif ps > ds:
            reward = 1.0
        elif ps == ds:
            reward = 0.0
        else:
            reward = -1.0

        return visited_states, actions, reward

    # -----------------------------
    #   Aktualizacja MC First Visit
    # -----------------------------
    def play_episode_and_update(self):
        visited_states, actions, reward = self.play_episode_exploring_starts()
        seen = set()  # First-visit

        for state, action in zip(visited_states, actions):
            if (state, action) in seen:
                continue
            seen.add((state, action))

            ps, dc, ua = state
            cur_n = self.n[ps, dc, ua, action]
            self.Q[ps, dc, ua, action] = (self.Q[ps, dc, ua, action]*cur_n + reward) / (cur_n + 1)
            self.n[ps, dc, ua, action] += 1

        return reward

    def train(self, episodes=200000, report_every=20000):
        for i in range(episodes):
            self.play_episode_and_update()
            if (i+1) % report_every == 0:
                print(f"[ES] Trained {i+1}/{episodes}")

    def evaluate_policy(self, print_n=False):
        from utils import print_policy
        print_policy(self.Q, self.n, title="Exploring Starts Policy", print_n=print_n)