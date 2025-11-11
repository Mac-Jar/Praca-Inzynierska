import random
import numpy as np
import cupy as cp

from Blackjack import Blackjack

class AI_Blackjack_CUDA:
    def __init__(self, epsilon=0.1, gamma=1.0, alpha=0.1):
        # Q[player_sum][dealer_card][usable_ace][action]
        # action: 0 = stand, 1 = hit

        self.Q = cp.zeros((32, 11, 2, 2))
        self.n = cp.zeros((32, 11, 2, 2))

        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.blackjackGame = Blackjack()

    def get_state(self):
        player_sum = self.blackjackGame.evaluate_hand(self.blackjackGame.player_cards)
        if player_sum > 21:
            player_sum = 22
        dealer_card = self.blackjackGame.dealer_cards[0].blackjack_card_value()
        dealer_card = min(max(1, dealer_card), 10)
        usable_ace = 0
        if any(c.value == "Ace" for c in self.blackjackGame.player_cards) and player_sum < 22:
            usable_ace = 1
        return player_sum, dealer_card, usable_ace

    def choose_action(self, state):
        player_sum, dealer_card, usable_ace = state
        # eksploracja
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        # pobieramy wiersz z GPU (view) i robimy mały transfer indeksów maksymalnych
        row = self.Q[player_sum, dealer_card, usable_ace]  # CuPy array length 2
        max_val = cp.max(row)
        # flatnonzero na GPU, potem przesyłamy mały wektor na CPU
        indices = cp.flatnonzero(row == max_val)
        indices_cpu = cp.asnumpy(indices)  # zazwyczaj bardzo krótki (<=2)
        return int(np.random.choice(indices_cpu))

    def choose_action_final(self, state):
        player_sum, dealer_card, usable_ace = state
        row = self.Q[player_sum, dealer_card, usable_ace]
        return int(int(cp.argmax(row).get()))

    def play_episode(self):
        game = self.blackjackGame
        game.initialize()
        game.deal_cards(shuffle=True)

        states = []
        actions = []

        state = self.get_state()
        action = self.choose_action(state)
        states.append(state)
        actions.append(action)

        while action != 0:
            if action == 1:
                game.player_hits()
                player_sum = game.evaluate_hand(game.player_cards)
                # jeśli przegrany natychmiast, dodajemy końcowy stan (opcjonalnie)
                if player_sum > 21:
                    # zarejestruj stan po biciu (może być przydatne)
                    state = self.get_state()
                    states.append(state)
                    actions.append(0)  # nic nie trzeba, ale zachowujemy parzystość
                    break
                # kontynuuj
                state = self.get_state()
                action = self.choose_action(state)
                states.append(state)
                actions.append(action)
            else:
                break

        reward = game.check_game_result()
        # WYKONAJ WEKTORYZOWANĄ AKTUALIZACJĘ Q I n NA GPU:
        # Przygotuj indeksy jako numpy arrays (mały transfer CPU->GPU)
        if len(states) == 0:
            return

        p_sums = np.array([s[0] for s in states], dtype=np.int32)
        d_cards = np.array([s[1] for s in states], dtype=np.int32)
        us_aces = np.array([s[2] for s in states], dtype=np.int32)
        acts = np.array(actions, dtype=np.int32)
        rewards = np.full_like(acts, reward, dtype=np.int32)  # ten sam reward dla wszystkich stanów w epizodzie

        # przenieś indeksy na GPU
        p_sums_gpu = cp.asarray(p_sums)
        d_cards_gpu = cp.asarray(d_cards)
        us_aces_gpu = cp.asarray(us_aces)
        acts_gpu = cp.asarray(acts)
        rewards_gpu = cp.asarray(rewards, dtype=cp.float32)

        # odczyt Q_old i n_old przez zaawansowane indeksowanie
        q_old = self.Q[p_sums_gpu, d_cards_gpu, us_aces_gpu, acts_gpu].astype(cp.float32)
        n_old = self.n[p_sums_gpu, d_cards_gpu, us_aces_gpu, acts_gpu].astype(cp.float32)

        # oblicz nowe wartości średniej: q_new = (q_old * n_old + reward) / (n_old + 1)
        q_new = (q_old * n_old + rewards_gpu) / (n_old + 1.0)
        n_new = (n_old + 1.0).astype(cp.int32)

        # zapis na GPU
        self.Q[p_sums_gpu, d_cards_gpu, us_aces_gpu, acts_gpu] = q_new
        self.n[p_sums_gpu, d_cards_gpu, us_aces_gpu, acts_gpu] = n_new

    def train(self, episodes=50000, log_every=5000):
        for i in range(episodes):
            self.play_episode()
            if (i + 1) % log_every == 0:
                print(f"Trenowanie... {i + 1}/{episodes}")

    def evaluate_policy(self, print_n=True):
        print("\n=== Strategia AI po treningu ===")
        # przeniesienie Q i n częściowo na CPU celem czytelnego wydruku
        Q_cpu = cp.asnumpy(self.Q)
        n_cpu = cp.asnumpy(self.n)
        for usable in [1, 0]:
            print(f"\nUsable Ace = {usable}")
            for dealer_card in range(1, 11):
                decisions = []
                for player_sum in range(12, 22):
                    action = int(np.argmax(Q_cpu[player_sum, dealer_card, usable]))
                    decision = "HIT" if action == 1 else "STAND"
                    decisions.append(f"{player_sum}:{decision}")
                print(f"Dealer {dealer_card}: " + ", ".join(decisions))

            print("\n--- STAND (akcja 0) ---")
            for player_sum in range(2, 22):
                row = [int(n_cpu[player_sum, dealer_card, usable, 0]) for dealer_card in range(2, 11)]
                print(f"{player_sum:2d}: " + " ".join(f"{val:6d}" for val in row))

            print("\n--- HIT (akcja 1) ---")
            for player_sum in range(2, 22):
                row = [int(n_cpu[player_sum, dealer_card, usable, 1]) for dealer_card in range(2, 11)]
                print(f"{player_sum:2d}: " + " ".join(f"{val:6d}" for val in row))