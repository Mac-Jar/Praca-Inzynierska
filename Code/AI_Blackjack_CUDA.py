import random
import numpy as np
import cupy as cp

from Blackjack import Blackjack


class AI_Blackjack_CUDA:
    """
    Równoległa symulacja epizodów z użyciem klasy Blackjack (CPU),
    a aktualizacje Q/n/suma_nagród wykonywane na GPU (CuPy).
    """
    def __init__(self, epsilon=0.1, gamma=1.0):
        # Wymiary: Q[player_sum 0..31][dealer_upcard 1..10][usable_ace 0/1][action 0/1]
        self.Q = cp.zeros((32, 11, 2, 2), dtype=cp.float32)
        self.n = cp.zeros((32, 11, 2, 2), dtype=cp.int32)
        # Przechowywanie sum nagród dla łatwego uśredniania (Q = R / n)
        self.R = cp.zeros((32, 11, 2, 2), dtype=cp.float32)

        self.epsilon = epsilon
        self.gamma = gamma

    # ----------------------------
    # Pomocnicze: stan z instancji Blackjack
    # ----------------------------
    @staticmethod
    def get_state_from_game(game: Blackjack):
        player_sum = game.evaluate_hand(game.player_cards)
        if player_sum > 21:
            player_sum = 22
        dealer_card = game.dealer_cards[0].blackjack_card_value()
        dealer_card = min(max(1, dealer_card), 10)
        usable_ace = 0
        if any(c.value == "Ace" for c in game.player_cards) and player_sum < 22:
            usable_ace = 1
        return player_sum, dealer_card, usable_ace

    # ----------------------------
    # Wybór akcji (ε-greedy) na GPU
    # ----------------------------
    def choose_action_batch(self, psum_cpu, upcard_cpu, usable_cpu):
        """
        psum_cpu, upcard_cpu, usable_cpu: ndarray int32 (aktywni gracze)
        Zwraca: ndarray int32 (0=STAND, 1=HIT)
        """
        # przerzuć indeksy na GPU
        psum = cp.asarray(psum_cpu, dtype=cp.int32)
        upcard = cp.asarray(upcard_cpu, dtype=cp.int32)
        usable = cp.asarray(usable_cpu, dtype=cp.int32)

        # Greedy: porównaj Q dla STAND/HIT
        q_stand = self.Q[psum, upcard, usable, 0]
        q_hit = self.Q[psum, upcard, usable, 1]
        greedy = (q_hit > q_stand).astype(cp.int32)  # 1 jeśli HIT lepszy, inaczej 0 (STAND)

        # Eksploracja
        rand_actions = cp.random.randint(0, 2, size=psum.shape[0], dtype=cp.int32)
        explore_mask = cp.random.random(size=psum.shape[0]) < self.epsilon

        actions = cp.where(explore_mask, rand_actions, greedy)
        return cp.asnumpy(actions)  # zwróć na CPU do sterowania środowiskiem

    # ----------------------------
    # Jedna paczka gier, lockstep
    # ----------------------------
    def play_batch(self, batch_size: int):
        """
        Uruchamia równolegle batch_size gier (CPU), zbiera trajektorie,
        i aktualizuje Q/R/n na GPU w jednej/dwóch dużych operacjach.
        """
        # Inicjalizacja wielu gier
        games = [Blackjack() for _ in range(batch_size)]
        for g in games:
            g.initialize()
            g.deal_cards(shuffle=True)

        # Bufory trajektorii per gra (CPU): listy trójek stanu i akcji
        traj_psum = [[] for _ in range(batch_size)]
        traj_upcard = [[] for _ in range(batch_size)]
        traj_usable = [[] for _ in range(batch_size)]
        traj_action = [[] for _ in range(batch_size)]

        # Maski gry zakończone
        done = np.zeros(batch_size, dtype=bool)

        # Pierwszy krok: zapisz stan i akcję dla wszystkich
        psum = np.empty(batch_size, dtype=np.int32)
        upcard = np.empty(batch_size, dtype=np.int32)
        usable = np.empty(batch_size, dtype=np.int32)

        for i, g in enumerate(games):
            s = self.get_state_from_game(g)
            psum[i], upcard[i], usable[i] = s

        # Pętla kroków gracza (lockstep)
        # Limit bezpieczeństwa kroków (np. maks 20 akcji na grę)
        for _ in range(20):
            active_idx = np.flatnonzero(~done)
            if active_idx.size == 0:
                break

            # Wybór akcji na GPU dla aktywnych
            actions = self.choose_action_batch(psum[active_idx], upcard[active_idx], usable[active_idx])

            # Zapisz stan + akcję do trajektorii
            for j, gi in enumerate(active_idx):
                traj_psum[gi].append(int(psum[gi]))
                traj_upcard[gi].append(int(upcard[gi]))
                traj_usable[gi].append(int(usable[gi]))
                traj_action[gi].append(int(actions[j]))

            # Wykonaj akcje w środowisku (CPU)
            # HIT → dobierz kartę; STAND → zakończ turę gracza
            for j, gi in enumerate(active_idx):
                if actions[j] == 1:  # HIT
                    games[gi].player_hits()
                    # sprawdź bust po HIT
                    player_sum = games[gi].evaluate_hand(games[gi].player_cards)
                    if player_sum > 21:
                        done[gi] = True
                else:  # STAND
                    done[gi] = True

            # Zaktualizuj stany dla niezakończonych (lub po HIT)
            for gi in active_idx:
                if not done[gi]:
                    s = self.get_state_from_game(games[gi])
                    psum[gi], upcard[gi], usable[gi] = s

        # Policzenie wyniku gry (CPU)
        rewards = np.empty(batch_size, dtype=np.float32)
        for i, g in enumerate(games):
            rewards[i] = float(g.check_game_result())

        # ----------------------------
        # Aktualizacja na GPU (wektorowo, per-krok z poprawnym reward per gra)
        # ----------------------------
        # Sklej wszystkie kroki ze wszystkich gier w jedną dużą paczkę indeksów
        all_psum = np.concatenate([np.array(traj_psum[i], dtype=np.int32) for i in range(batch_size)], axis=0)
        all_upcard = np.concatenate([np.array(traj_upcard[i], dtype=np.int32) for i in range(batch_size)], axis=0)
        all_usable = np.concatenate([np.array(traj_usable[i], dtype=np.int32) for i in range(batch_size)], axis=0)
        all_action = np.concatenate([np.array(traj_action[i], dtype=np.int32) for i in range(batch_size)], axis=0)

        # Zbuduj wektor nagród krokowych przez broadcast per gra:
        # dla każdego kroku weź reward odpowiadającej gry
        step_rewards = np.concatenate([
            np.full(len(traj_action[i]), rewards[i], dtype=np.float32)
            for i in range(batch_size)
        ], axis=0)

        # Przerzuć indeksy i nagrody na GPU
        ps_gpu = cp.asarray(all_psum, dtype=cp.int32)
        dc_gpu = cp.asarray(all_upcard, dtype=cp.int32)
        ua_gpu = cp.asarray(all_usable, dtype=cp.int32)
        ac_gpu = cp.asarray(all_action, dtype=cp.int32)
        rw_gpu = cp.asarray(step_rewards, dtype=cp.float32)

        # Oblicz liniowe indeksy dla tablicy o kształcie (32, 11, 2, 2)
        S, D, U, A = 32, 11, 2, 2
        lin_idx = (((ps_gpu * D) + dc_gpu) * U + ua_gpu) * A + ac_gpu  # (T,)

        # Suma nagród per indeks (scatter-add) i liczność odwiedzin
        total_size = S * D * U * A
        sum_rewards = cp.bincount(lin_idx, weights=rw_gpu, minlength=total_size)  # (total_size,)
        visit_counts = cp.bincount(lin_idx, minlength=total_size)  # (total_size,)

        # Dodaj do R i n po przekształceniu do 4D
        self.R += sum_rewards.reshape(S, D, U, A).astype(cp.float32)
        self.n += visit_counts.reshape(S, D, U, A).astype(cp.int32)

        # Zaktualizuj Q = R / n tam, gdzie n > 0
        nonzero = self.n > 0
        self.Q[nonzero] = (self.R[nonzero] / self.n[nonzero].astype(cp.float32)).astype(cp.float32)

    # ----------------------------
    # Trening: epizody w paczkach
    # ----------------------------
    def train(self, episodes=50000, batch_size=1024, log_every=5000):
        batches = episodes // batch_size
        for b in range(batches):
            self.play_batch(batch_size)
            done_eps = (b + 1) * batch_size
            if done_eps % log_every == 0:
                q_mean = float(self.Q.mean())
                print(f"Trenowanie... {done_eps}/{episodes} | Q_mean={q_mean:.4f}")

    # ----------------------------
    # Ewaluacja i drukowanie polityki
    # ----------------------------
    def evaluate_policy(self, print_n=True):
        print("\n=== Strategia AI po treningu ===")
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
            if not print_n:
                continue
            print("\n--- STAND (akcja 0) ---")
            for player_sum in range(2, 22):
                row = [int(n_cpu[player_sum, dealer_card, usable, 0]) for dealer_card in range(2, 11)]
                print(f"{player_sum:2d}: " + " ".join(f"{val:6d}" for val in row))

            print("\n--- HIT (akcja 1) ---")
            for player_sum in range(2, 22):
                row = [int(n_cpu[player_sum, dealer_card, usable, 1]) for dealer_card in range(2, 11)]
                print(f"{player_sum:2d}: " + " ".join(f"{val:6d}" for val in row))
