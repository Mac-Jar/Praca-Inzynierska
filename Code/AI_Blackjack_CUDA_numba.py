# ai_gpu.py
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
import time
from utils import *
# shape constants
PS_MAX = 32
DC_MAX = 11
UA_MAX = 2
ACTIONS = 2

# device helpers
@cuda.jit(device=True, inline=True)
def evaluate_hand_gpu(cards_vals, n):
    total = 0
    aces = 0
    for i in range(n):
        v = cards_vals[i]
        total += v
        if v == 11:
            aces += 1
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total

@cuda.jit(device=True, inline=True)
def draw_card(rng_states, tid):
    # use xoroshiro uniform generator helper
    return int(xoroshiro128p_uniform_float32(rng_states, tid) * 13) + 2  # gives 2..14, we'll map >=11 appropriately

@cuda.jit(device=True, inline=True)
def map_card_value(v):
    # v in 2..14 where 11-14 -> 10 (J,Q,K,10?), Ace -> treat as 11 when v==14
    if v >= 11 and v <= 13:
        return 10
    if v == 14:
        return 11
    return v

@cuda.jit
def play_episodes_kernel(dev_Q, epsilon, rng_states,
                         dev_sum_rewards, dev_counts, episodes_per_thread):
    tid = cuda.grid(1)
    n_threads = cuda.gridsize(1)
    if tid >= rng_states.shape[0]:
        return

    # local buffers
    visited_states = cuda.local.array(shape=(60,), dtype=np.int32)  # triplets, max steps ~20 -> 60 ints
    visited_actions = cuda.local.array(shape=(20,), dtype=np.int32)

    for ep in range(episodes_per_thread):
        # init small local "deckless" game using RNG
        # draw initial cards
        pcards = cuda.local.array(12, dtype=np.int32)
        dcards = cuda.local.array(12, dtype=np.int32)
        p_n = 0
        d_n = 0

        # draw cards (mapping)
        v = draw_card(rng_states, tid)
        pcards[p_n] = map_card_value(v)
        p_n += 1
        v = draw_card(rng_states, tid)
        pcards[p_n] = map_card_value(v)
        p_n += 1

        v = draw_card(rng_states, tid)
        dcards[d_n] = map_card_value(v)
        d_n += 1
        v = draw_card(rng_states, tid)
        dcards[d_n] = map_card_value(v)
        d_n += 1

        # player loop
        step = 0
        v_idx = 0
        while True:
            p_sum = evaluate_hand_gpu(pcards, p_n)
            ps_idx = p_sum if p_sum <= 31 else 31
            if p_sum > 21:
                ps_idx = 22
            dealer_up = dcards[0]
            if dealer_up > 10:
                dealer_up = 10
            usable = 0
            if p_sum <= 21:
                for i in range(p_n):
                    if pcards[i] == 11:
                        usable = 1
                        break

            # epsilon-greedy using dev_Q
            r = xoroshiro128p_uniform_float32(rng_states, tid)
            if r < epsilon:
                a = 1 if xoroshiro128p_uniform_float32(rng_states, tid) < 0.5 else 0
            else:
                q0 = dev_Q[ps_idx, dealer_up, usable, 0]
                q1 = dev_Q[ps_idx, dealer_up, usable, 1]
                a = 0 if q0 >= q1 else 1

            # store
            if step < 20:
                visited_states[v_idx]   = ps_idx
                visited_states[v_idx+1] = dealer_up
                visited_states[v_idx+2] = usable
                visited_actions[step] = a
                v_idx += 3
            step += 1

            # action effect
            if a == 1:
                # hit
                v = draw_card(rng_states, tid)
                pcards[p_n] = map_card_value(v)
                p_n += 1
                if evaluate_hand_gpu(pcards, p_n) > 21:
                    break
            else:
                break

            if step >= 20:
                break

        # determine reward
        p_sum = evaluate_hand_gpu(pcards, p_n)
        if p_sum > 21:
            reward = -1.0
        else:
            # dealer plays
            while evaluate_hand_gpu(dcards, d_n) < 17:
                v = draw_card(rng_states, tid)
                dcards[d_n] = map_card_value(v)
                d_n += 1
            d_sum = evaluate_hand_gpu(dcards, d_n)
            if d_sum > 21:
                reward = 1.0
            elif d_sum > p_sum:
                reward = -1.0
            elif d_sum < p_sum:
                reward = 1.0
            else:
                reward = 0.0

        # atomically add reward & counts for all visited (s,a)
        visits = step if step < 20 else 20
        base = 0
        for i in range(visits):
            ps_idx = visited_states[base]
            dc = visited_states[base+1]
            ua = visited_states[base+2]
            a = visited_actions[i]
            # atomic add
            cuda.atomic.add(dev_sum_rewards, (ps_idx, dc, ua, a), reward)
            cuda.atomic.add(dev_counts, (ps_idx, dc, ua, a), 1)
            base += 3

# rest of ai_gpu.py (wrapper)
class AI_Blackjack_GPU:
    def __init__(self, epsilon=0.1, device_threads=256, seed=1234):
        self.epsilon = epsilon
        self.shape = (PS_MAX, DC_MAX, UA_MAX, ACTIONS)
        self.sum_rewards = np.zeros(self.shape, dtype=np.float32)
        self.counts = np.zeros(self.shape, dtype=np.int32)
        self.Q = np.ones(self.shape, dtype=np.float32)
        self.Q[:,:,:,1]=2
        # device arrays
        self.d_sum_rewards = cuda.to_device(self.sum_rewards)
        self.d_counts = cuda.to_device(self.counts)
        self.d_Q = cuda.to_device(self.Q)

        # RNG states
        self.device_threads = device_threads
        self.seed = seed
        # choose number of threads (grid) later per-run
        self.rng_states = None

    def _ensure_rng(self, n_threads):
        if self.rng_states is None or self.rng_states.size != n_threads:
            self.rng_states = create_xoroshiro128p_states(n_threads, seed=self.seed)

    def train(self, episodes=200000, batch_size=50000, threads_per_block=256):
        """
        episodes: total episodes to run
        batch_episodes: how many episodes to simulate per host-device round
        """
        remaining = episodes
        while remaining > 0:
            cur_batch = min(remaining, batch_size)
            # compute threads and ep_per_thread
            # pick n_threads = blocks * threads_per_block
            n_threads = 1024  # choose baseline threads; can be tuned
            blocks = max(1, (n_threads + threads_per_block - 1)//threads_per_block)
            n_threads = blocks * threads_per_block
            ep_per_thread = max(1, cur_batch // n_threads)
            # ensure rng
            self._ensure_rng(n_threads)

            # reset device accumulators
            cuda.to_device(np.zeros(self.shape, dtype=np.float32), to=self.d_sum_rewards)
            cuda.to_device(np.zeros(self.shape, dtype=np.int32), to=self.d_counts)

            # copy current Q to device
            self.d_Q.copy_to_device(self.Q.astype(np.float32))

            # run kernel
            play_episodes_kernel[blocks, threads_per_block](self.d_Q, np.float32(self.epsilon),
                                                            self.rng_states,
                                                            self.d_sum_rewards,
                                                            self.d_counts,
                                                            np.int32(ep_per_thread))
            cuda.synchronize()

            # pull sums and counts
            sum_rewards = self.d_sum_rewards.copy_to_host()
            counts = self.d_counts.copy_to_host()

            # aggregate into host arrays
            # convert to float64 for stability in averaging
            mask = counts > 0
            self.sum_rewards[mask] += sum_rewards[mask].astype(np.float64)
            self.counts[mask] += counts[mask]

            # recompute Q where counts>0
            mask2 = self.counts > 0
            self.Q[mask2] = (self.sum_rewards[mask2] / self.counts[mask2]).astype(np.float32)

            remaining -= cur_batch
            print(f"[GPU] Completed batch of {cur_batch} episodes. Remaining: {remaining}")

    def get_greedy_action(self, state):
        player_sum, dealer_card, usable_ace = state
        return int(np.argmax(self.Q[player_sum, dealer_card, usable_ace]))

    def evaluate_policy(self, print_n=False):
        print_policy(self.Q, getattr(self, 'counts', None), title="GPU Policy", print_n=print_n)


# ai_gpu.py
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math
import time
from utils import *
# shape constants
PS_MAX = 32
DC_MAX = 11
UA_MAX = 2
ACTIONS = 2

# device helpers
@cuda.jit(device=True, inline=True)
def evaluate_hand_gpu(cards_vals, n):
    total = 0
    aces = 0
    for i in range(n):
        v = cards_vals[i]
        total += v
        if v == 11:
            aces += 1
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total

@cuda.jit(device=True, inline=True)
def draw_card(rng_states, tid):
    # use xoroshiro uniform generator helper
    return int(xoroshiro128p_uniform_float32(rng_states, tid) * 13) + 2  # gives 2..14, we'll map >=11 appropriately

@cuda.jit(device=True, inline=True)
def map_card_value(v):
    # v in 2..14 where 11-14 -> 10 (J,Q,K,10?), Ace -> treat as 11 when v==14
    if v >= 11 and v <= 13:
        return 10
    if v == 14:
        return 11
    return v

@cuda.jit
def play_episodes_es_kernel(dev_Q,
                            rng_states,
                            dev_sum_rewards,
                            dev_counts,
                            episodes_per_thread):
    tid = cuda.grid(1)
    if tid >= rng_states.shape[0]:
        return

    visited_states = cuda.local.array(60, dtype=np.int32)
    visited_actions = cuda.local.array(20, dtype=np.int32)

    for ep in range(episodes_per_thread):

        # -----------------------------
        # 1. Exploring Start: losowy stan
        # -----------------------------
        ps = int(xoroshiro128p_uniform_float32(rng_states, tid) * 10) + 12   # 12..21
        dealer_up = int(xoroshiro128p_uniform_float32(rng_states, tid) * 10) + 1  # 1..10
        usable = 1 if xoroshiro128p_uniform_float32(rng_states, tid) < 0.5 else 0

        # skonstruuj rękę gracza tak, aby dawała player_sum == ps
        pcards = cuda.local.array(12, dtype=np.int32)
        p_n = 0
        rest = ps

        if usable == 1:
            pcards[p_n] = 11
            p_n += 1
            rest -= 11

        while rest > 0:
            card = int(xoroshiro128p_uniform_float32(rng_states, tid) * 9) + 2  # 2..10
            if card > rest:
                card = rest
            pcards[p_n] = card
            p_n += 1
            rest -= card

        # dealer ma kartę odkrytą + jedną losową
        dcards = cuda.local.array(12, dtype=np.int32)
        d_n = 0
        dcards[d_n] = dealer_up
        d_n += 1
        v = draw_card(rng_states, tid)
        dcards[d_n] = map_card_value(v)
        d_n += 1

        # -----------------------------
        # 2. Pierwsza akcja = losowa (exploring start)
        # -----------------------------
        a = 1 if xoroshiro128p_uniform_float32(rng_states, tid) < 0.5 else 0

        step = 0
        v_idx = 0

        visited_states[v_idx] = ps
        visited_states[v_idx+1] = dealer_up
        visited_states[v_idx+2] = usable
        visited_actions[step] = a
        v_idx += 3
        step += 1

        # efekt akcji
        if a == 1:
            v = draw_card(rng_states, tid)
            pcards[p_n] = map_card_value(v)
            p_n += 1
            if evaluate_hand_gpu(pcards, p_n) > 21:
                # zas bust po exploring start
                reward = -1.0
                # zapisujemy wszystkie odwiedzone stany/akcje
                base = 0
                for i in range(step):
                    ps2 = visited_states[base]
                    dc2 = visited_states[base+1]
                    ua2 = visited_states[base+2]
                    aa = visited_actions[i]
                    cuda.atomic.add(dev_sum_rewards, (ps2, dc2, ua2, aa), reward)
                    cuda.atomic.add(dev_counts, (ps2, dc2, ua2, aa), 1)
                    base += 3
                continue

        # -----------------------------
        # 3. Reszta gry — greedy( Q )
        # -----------------------------
        while True:
            p_sum = evaluate_hand_gpu(pcards, p_n)
            if p_sum > 21:
                break

            if p_sum > 21:
                ps_idx = 22
            else:
                ps_idx = p_sum

            dealer_idx = dcards[0]
            if dealer_idx > 10:
                dealer_idx = 10

            usable_idx = 0
            if p_sum <= 21:
                for i in range(p_n):
                    if pcards[i] == 11:
                        usable_idx = 1
                        break

            # greedy
            q0 = dev_Q[ps_idx, dealer_idx, usable_idx, 0]
            q1 = dev_Q[ps_idx, dealer_idx, usable_idx, 1]
            a = 0 if q0 >= q1 else 1

            if step < 20:
                visited_states[v_idx] = ps_idx
                visited_states[v_idx+1] = dealer_idx
                visited_states[v_idx+2] = usable_idx
                visited_actions[step] = a
                v_idx += 3
            step += 1

            if a == 1:
                v = draw_card(rng_states, tid)
                pcards[p_n] = map_card_value(v)
                p_n += 1
                if evaluate_hand_gpu(pcards, p_n) > 21:
                    break
            else:
                break

            if step >= 20:
                break

        # -----------------------------
        # 4. Dealer + reward
        # -----------------------------
        p_sum = evaluate_hand_gpu(pcards, p_n)
        if p_sum > 21:
            reward = -1.0
        else:
            while evaluate_hand_gpu(dcards, d_n) < 17:
                v = draw_card(rng_states, tid)
                dcards[d_n] = map_card_value(v)
                d_n += 1
            d_sum = evaluate_hand_gpu(dcards, d_n)
            if d_sum > 21:
                reward = 1.0
            elif d_sum > p_sum:
                reward = -1.0
            elif d_sum < p_sum:
                reward = 1.0
            else:
                reward = 0.0

        # -----------------------------
        # 5. atomic add MC update
        # -----------------------------
        visits = step if step < 20 else 20
        base = 0
        for i in range(visits):
            ps2 = visited_states[base]
            dc2 = visited_states[base+1]
            ua2 = visited_states[base+2]
            aa = visited_actions[i]
            cuda.atomic.add(dev_sum_rewards, (ps2, dc2, ua2, aa), reward)
            cuda.atomic.add(dev_counts, (ps2, dc2, ua2, aa), 1)
            base += 3


# rest of ai_gpu.py (wrapper)
class AI_Blackjack_GPU_ES:
    def __init__(self, epsilon=0.1, device_threads=256, seed=1234):
        self.epsilon = epsilon
        self.shape = (PS_MAX, DC_MAX, UA_MAX, ACTIONS)
        self.sum_rewards = np.zeros(self.shape, dtype=np.float32)
        self.counts = np.zeros(self.shape, dtype=np.int32)
        self.Q = np.ones(self.shape, dtype=np.float32)
        self.Q[:,:,:,1]=2
        # device arrays
        self.d_sum_rewards = cuda.to_device(self.sum_rewards)
        self.d_counts = cuda.to_device(self.counts)
        self.d_Q = cuda.to_device(self.Q)

        # RNG states
        self.device_threads = device_threads
        self.seed = seed
        # choose number of threads (grid) later per-run
        self.rng_states = None

    def _ensure_rng(self, n_threads):
        if self.rng_states is None or self.rng_states.size != n_threads:
            self.rng_states = create_xoroshiro128p_states(n_threads, seed=self.seed)

    def train(self, episodes=200000, batch_size=50000, threads_per_block=256):
        """
        episodes: total episodes to run
        batch_episodes: how many episodes to simulate per host-device round
        """
        remaining = episodes
        while remaining > 0:
            cur_batch = min(remaining, batch_size)
            # compute threads and ep_per_thread
            # pick n_threads = blocks * threads_per_block
            n_threads = 1024  # choose baseline threads; can be tuned
            blocks = max(1, (n_threads + threads_per_block - 1)//threads_per_block)
            n_threads = blocks * threads_per_block
            ep_per_thread = max(1, cur_batch // n_threads)
            # ensure rng
            self._ensure_rng(n_threads)

            # reset device accumulators
            cuda.to_device(np.zeros(self.shape, dtype=np.float32), to=self.d_sum_rewards)
            cuda.to_device(np.zeros(self.shape, dtype=np.int32), to=self.d_counts)

            # copy current Q to device
            self.d_Q.copy_to_device(self.Q.astype(np.float32))

            # run kernel
            play_episodes_es_kernel[blocks, threads_per_block](self.d_Q,
                                                            self.rng_states,
                                                            self.d_sum_rewards,
                                                            self.d_counts,
                                                            np.int32(ep_per_thread))
            cuda.synchronize()

            # pull sums and counts
            sum_rewards = self.d_sum_rewards.copy_to_host()
            counts = self.d_counts.copy_to_host()

            # aggregate into host arrays
            # convert to float64 for stability in averaging
            mask = counts > 0
            self.sum_rewards[mask] += sum_rewards[mask].astype(np.float64)
            self.counts[mask] += counts[mask]

            # recompute Q where counts>0
            mask2 = self.counts > 0
            self.Q[mask2] = (self.sum_rewards[mask2] / self.counts[mask2]).astype(np.float32)

            remaining -= cur_batch
            print(f"[GPU] Completed batch of {cur_batch} episodes. Remaining: {remaining}")

    def get_greedy_action(self, state):
        player_sum, dealer_card, usable_ace = state
        return int(np.argmax(self.Q[player_sum, dealer_card, usable_ace]))

    def evaluate_policy(self, print_n=False):
        print_policy(self.Q, getattr(self, 'counts', None), title="GPU Policy", print_n=print_n)
