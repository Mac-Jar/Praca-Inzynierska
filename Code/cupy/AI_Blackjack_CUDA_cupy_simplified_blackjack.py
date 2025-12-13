import numpy as np
import cupy as cp
from utils import print_policy

class AI_Blackjack_CuPy:
    """
    AI Blackjack w stylu Sutton & Barto, implementacja z CuPy.
    W pełni kompatybilne z CPU i Numba GPU.
    """
    def __init__(self, epsilon=0.1, gamma=1.0):
        # Q[player_sum, dealer_card, usable_ace, action]
        self.Q = cp.zeros((32, 11, 2, 2), dtype=cp.float32)
        self.n = cp.zeros((32, 11, 2, 2), dtype=cp.int32)
        self.R = cp.zeros((32, 11, 2, 2), dtype=cp.float32)
        self.epsilon = epsilon
        self.gamma = gamma

    # ----------------------------
    # Stany i logika gry
    # ----------------------------
    @staticmethod
    def get_initial_state(batch_size):
        """
        Losowe początkowe sumy gracza i karty dealera
        player_sum: 12-21 (typowy zakres)
        dealer_card: 1-10
        usable_ace: 0/1
        """
        psum = np.random.randint(12, 22, size=batch_size, dtype=np.int32)
        upcard = np.random.randint(1, 11, size=batch_size, dtype=np.int32)
        usable = np.random.randint(0, 2, size=batch_size, dtype=np.int32)
        return psum, upcard, usable

    @staticmethod
    def draw_card(batch_size):
        """
        Losowa karta 1-10 (AS liczymy jako 11)
        """
        return np.random.randint(1, 12, size=batch_size, dtype=np.int32)

    @staticmethod
    def update_sum(player_sum, card, usable_ace):
        """
        Dodaje kartę i uwzględnia usable ace
        """
        psum = player_sum + card
        ua = usable_ace.copy()
        # Jeśli dodana karta to As
        ua[card == 11] = 1
        # Jeśli przebicie i mamy usable ace → odejmij 10
        over = (psum > 21) & (ua == 1)
        psum[over] -= 10
        ua[over] = 0
        return psum, ua

    # ----------------------------
    # Wybór akcji ε-greedy
    # ----------------------------
    def choose_action_batch(self, psum_cpu, upcard_cpu, usable_cpu):
        psum = cp.asarray(psum_cpu, dtype=cp.int32)
        upcard = cp.asarray(upcard_cpu, dtype=cp.int32)
        usable = cp.asarray(usable_cpu, dtype=cp.int32)

        q_hit = self.Q[psum, upcard, usable, 1]
        q_stand = self.Q[psum, upcard, usable, 0]

        greedy = (q_hit > q_stand).astype(cp.int32)
        rand_actions = cp.random.randint(0, 2, size=psum.shape[0], dtype=cp.int32)
        explore_mask = cp.random.random(size=psum.shape[0]) < self.epsilon
        actions = cp.where(explore_mask, rand_actions, greedy)
        return cp.asnumpy(actions)

    # ----------------------------
    # Symulacja batch epizodów
    # ----------------------------
    def play_batch(self, batch_size=1024, max_steps=20):
        # Początkowe stany
        psum, upcard, usable = self.get_initial_state(batch_size)
        done = np.zeros(batch_size, dtype=bool)

        # Trajektorie
        traj_psum = [[] for _ in range(batch_size)]
        traj_upcard = [[] for _ in range(batch_size)]
        traj_usable = [[] for _ in range(batch_size)]
        traj_action = [[] for _ in range(batch_size)]

        for step in range(max_steps):
            active_idx = np.flatnonzero(~done)
            if active_idx.size == 0:
                break

            actions = self.choose_action_batch(psum[active_idx],
                                               upcard[active_idx],
                                               usable[active_idx])

            # Zapis trajektorii
            for j, gi in enumerate(active_idx):
                traj_psum[gi].append(int(psum[gi]))
                traj_upcard[gi].append(int(upcard[gi]))
                traj_usable[gi].append(int(usable[gi]))
                traj_action[gi].append(int(actions[j]))

            # Wykonanie akcji HIT/STAND
            draw = self.draw_card(active_idx.size)
            for j, gi in enumerate(active_idx):
                if actions[j] == 1:  # HIT
                    psum[gi], usable[gi] = self.update_sum(np.array([psum[gi]]),
                                                           np.array([draw[j]]),
                                                           np.array([usable[gi]]))
                    psum[gi] = int(psum[gi])
                    usable[gi] = int(usable[gi])
                    if psum[gi] > 21:
                        done[gi] = True
                else:  # STAND
                    done[gi] = True

        # Obliczenie wyników (dealer policy hit <17)
        rewards = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            dealer_sum = np.random.randint(1, 11)  # początkowa karta dealera
            dealer_ua = 0
            if dealer_sum == 11:
                dealer_ua = 1
            player_sum_i = psum[i]
            # dealer hits do 17
            while dealer_sum < 17:
                card = np.random.randint(1, 12)
                dealer_sum += card
                if card == 11:
                    dealer_ua = 1
                if dealer_sum > 21 and dealer_ua == 1:
                    dealer_sum -= 10
                    dealer_ua = 0
            # reward
            if player_sum_i > 21:
                rewards[i] = -1
            elif dealer_sum > 21 or player_sum_i > dealer_sum:
                rewards[i] = 1
            elif player_sum_i == dealer_sum:
                rewards[i] = 0
            else:
                rewards[i] = -1

        # ----------------------------
        # Aktualizacja Q/R/n na GPU
        # ----------------------------
        all_psum = np.concatenate([np.array(traj_psum[i], dtype=np.int32) for i in range(batch_size)])
        all_upcard = np.concatenate([np.array(traj_upcard[i], dtype=np.int32) for i in range(batch_size)])
        all_usable = np.concatenate([np.array(traj_usable[i], dtype=np.int32) for i in range(batch_size)])
        all_action = np.concatenate([np.array(traj_action[i], dtype=np.int32) for i in range(batch_size)])
        step_rewards = np.concatenate([np.full(len(traj_action[i]), rewards[i], dtype=np.float32)
                                       for i in range(batch_size)])

        ps_gpu = cp.asarray(all_psum, dtype=cp.int32)
        dc_gpu = cp.asarray(all_upcard, dtype=cp.int32)
        ua_gpu = cp.asarray(all_usable, dtype=cp.int32)
        ac_gpu = cp.asarray(all_action, dtype=cp.int32)
        rw_gpu = cp.asarray(step_rewards, dtype=cp.float32)

        # liniowe indeksy
        S, D, U, A = 32, 11, 2, 2
        lin_idx = (((ps_gpu * D) + dc_gpu) * U + ua_gpu) * A + ac_gpu

        total_size = S * D * U * A
        sum_rewards = cp.bincount(lin_idx, weights=rw_gpu, minlength=total_size)
        visit_counts = cp.bincount(lin_idx, minlength=total_size)

        self.R += sum_rewards.reshape(S, D, U, A).astype(cp.float32)
        self.n += visit_counts.reshape(S, D, U, A).astype(cp.int32)
        nonzero = self.n > 0
        self.Q[nonzero] = self.R[nonzero] / self.n[nonzero].astype(cp.float32)

    # ----------------------------
    # Trening
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
    # Wyświetlenie polityki
    # ----------------------------
    def evaluate_policy(self, print_n=True):
        print_policy(self.Q, self.n, title="CuPy AI Policy", print_n=print_n)