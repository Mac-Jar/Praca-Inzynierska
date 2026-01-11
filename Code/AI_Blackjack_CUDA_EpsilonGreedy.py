# ai_gpu.py
import numpy as np
from numba import cuda, float32, int32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32,xoroshiro128p_uniform_float64
from numba.types import float64, int64
import math
import time
from utils import *
# shape constants
PS_MAX = 32
DC_MAX = 11
UA_MAX = 2
ACTIONS = 2
TOTAL_STATES = PS_MAX * DC_MAX * UA_MAX * ACTIONS

def map_card_value_host(v):
    if 11 <= v <= 13:
        return 10
    if v == 14:
        return 11
    return v
# device helpers
@cuda.jit(device=True, inline=True)
def sum_and_usable_ace(cards_vals, n):
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
    usable = 1 if aces > 0 else 0
    return total, usable

@cuda.jit
def zero_array(arr):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] = 0

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
    return int(xoroshiro128p_uniform_float64(rng_states, tid) * 13) + 2  # gives 2..14, we'll map >=11 appropriately

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
    # --- Shared memory: 1D tablice ---
    block_sum = cuda.shared.array( shape=TOTAL_STATES, dtype=float64 )
    block_count = cuda.shared.array( shape=TOTAL_STATES, dtype=int64 )
    gid = cuda.grid(1)  # globalny indeks wątku
    # --- Zerowanie shared memory ---
    tid = cuda.threadIdx.x # lokalny indeks w bloku (do shared)
    block_size = cuda.blockDim.x
    active = gid < rng_states.shape[0]
    #total_size = PS_MAX * DC_MAX * UA_MAX * ACTIONS

    for i in range(tid, TOTAL_STATES, block_size):
        block_sum[i] = 0.0
        block_count[i] = 0

    cuda.syncthreads()
    if active:

    # local buffers
        visited_states = cuda.local.array(shape=(60,), dtype=np.int64)  # triplets, max steps ~20 -> 60 ints
        visited_actions = cuda.local.array(shape=(20,), dtype=np.int64)

        for ep in range(episodes_per_thread):
            # init small local "deckless" game using RNG
            # draw initial cards
            pcards = cuda.local.array(12, dtype=np.int64)
            dcards = cuda.local.array(12, dtype=np.int64)
            p_n = 0
            d_n = 0

            # draw cards (mapping)
            v = draw_card(rng_states, gid)
            pcards[p_n] = map_card_value(v)
            p_n += 1
            v = draw_card(rng_states, gid)
            pcards[p_n] = map_card_value(v)
            p_n += 1

            v = draw_card(rng_states, gid)
            dcards[d_n] = map_card_value(v)
            d_n += 1
            v = draw_card(rng_states, gid)
            dcards[d_n] = map_card_value(v)
            d_n += 1

            # player loop
            step = 0
            v_idx = 0
            while True:
                # p_sum = evaluate_hand_gpu(pcards, p_n)
                p_sum, usable = sum_and_usable_ace(pcards, p_n)
                ps_idx = p_sum if p_sum <= 31 else 31
                if p_sum > 21:
                    ps_idx = 22
                dealer_up = dcards[0]
                if dealer_up == 11:
                    dealer_up = 1  # indeks 1 oznacza asa
                elif dealer_up > 10:
                    dealer_up = 10
                # usable = 0
                # if p_sum <= 21:
                #     for i in range(p_n):
                #         if pcards[i] == 11:
                #             usable = 1
                #             break

                # epsilon-greedy using dev_Q
                r = xoroshiro128p_uniform_float64(rng_states, gid)

                if p_sum >= 21:
                    a = 0  # STAND at 21+

                elif r < epsilon:
                    a = 1 if xoroshiro128p_uniform_float64(rng_states, gid) < 0.5 else 0

                else:
                    q0 = dev_Q[ps_idx, dealer_up, usable, 0]  # STAND
                    q1 = dev_Q[ps_idx, dealer_up, usable, 1]  # HIT
                    if q0 > q1:
                        a = 0
                    elif q1 > q0:
                        a = 1
                    else:
                        a = 1 if xoroshiro128p_uniform_float64(rng_states, gid) < 0.5 else 0  # unbiased tie-break

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
                    v = draw_card(rng_states, gid)
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
                    v = draw_card(rng_states, gid)
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
                # cuda.atomic.add(dev_sum_rewards, (ps_idx, dc, ua, a), reward)
                # cuda.atomic.add(dev_counts, (ps_idx, dc, ua, a), 1)
                idx = ((ps_idx * DC_MAX + dc) * UA_MAX + ua) * ACTIONS + a
                cuda.atomic.add(block_sum, idx, reward)
                cuda.atomic.add(block_count, idx, 1)
                base += 3
    cuda.syncthreads()
    for i in range(tid, TOTAL_STATES, block_size):
        val_sum = block_sum[i]
        val_cnt = block_count[i]
        if val_cnt > 0:
            ps = i // (DC_MAX * UA_MAX * ACTIONS)
            rem = i % (DC_MAX * UA_MAX * ACTIONS)
            dc = rem // (UA_MAX * ACTIONS)
            rem = rem % (UA_MAX * ACTIONS)
            ua = rem // ACTIONS
            a = rem % ACTIONS

            cuda.atomic.add(dev_sum_rewards, (ps, dc, ua, a), val_sum)
            cuda.atomic.add(dev_counts, (ps, dc, ua, a), val_cnt)


def compute_launch_config(cur_batch, threads_per_block=256, max_threads=65536):
    n_threads = min(cur_batch, max_threads)
    blocks = (n_threads + threads_per_block - 1) // threads_per_block
    n_threads = blocks * threads_per_block
    ep_per_thread = max(1, cur_batch // n_threads)
    return blocks, threads_per_block, ep_per_thread
# rest of ai_gpu.py (wrapper)
class AI_Blackjack_GPU:
    def __init__(self, epsilon=0.1, device_threads=256, seed=1234):
        self.epsilon = epsilon
        self.shape = (PS_MAX, DC_MAX, UA_MAX, ACTIONS)
        self.sum_rewards = np.zeros(self.shape, dtype=np.float64)
        self.counts = np.zeros(self.shape, dtype=np.int64)
        self.Q = np.zeros(self.shape, dtype=np.float64)
        #self.Q[:,:,:,1]=2
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

    def train(self, episodes=200_000, target_episodes_per_thread=32,print_progress=True):
        """
        Epsilon-greedy training loop.
        episodes: total episodes to run
        target_episodes_per_thread: ile epizodów średnio przypada na wątek
        """
        batch_size = min(10**9,episodes // 100)
        threads_per_block = 256
        remaining = episodes

        history = make_training_history()
        batch_counter = 0
        log_every = 1  # loguj co batch; ustaw np. 10 żeby logować rzadziej
        # jeśli chcesz porównywać do optymalnej strategii, przygotuj tabele:
        optimal_hard, optimal_soft = generate_optimal_tables()

        while remaining > 0:
            cur_batch = min(remaining, batch_size)

            # --- Dobór liczby wątków ---
            n_threads = max(8192, cur_batch // target_episodes_per_thread)
            blocks = (n_threads + threads_per_block - 1) // threads_per_block
            n_threads = blocks * threads_per_block
            ep_per_thread = max(1, cur_batch // n_threads)

            # --- RNG ---
            self._ensure_rng(n_threads)

            # --- reset sum/count na GPU ---
            threads_zero = 256
            blocks_zero = max(1024, (self.d_sum_rewards.size + threads_zero - 1) // threads_zero)
            zero_array[blocks_zero, threads_zero](self.d_sum_rewards)

            blocks_zero = max(1024, (self.d_counts.size + threads_zero - 1) // threads_zero)
            zero_array[blocks_zero, threads_zero](self.d_counts)
            cuda.synchronize()

            # --- kopiowanie Q na GPU ---
            self.d_Q.copy_to_device(self.Q)

            # --- uruchomienie kernela epsilon-greedy ---
            play_episodes_kernel[blocks, threads_per_block](
                self.d_Q,
                np.float64(self.epsilon),
                self.rng_states,
                self.d_sum_rewards,
                self.d_counts,
                np.int64(ep_per_thread)
            )
            cuda.synchronize()

            # --- pobranie wyników ---
            sum_rewards = self.d_sum_rewards.copy_to_host()
            counts = self.d_counts.copy_to_host()

            # --- agregacja ---
            mask = counts > 0
            self.sum_rewards[mask] += sum_rewards[mask]
            self.counts[mask] += counts[mask]

            mask2 = self.counts > 0
            self.Q[mask2] = (self.sum_rewards[mask2] / self.counts[mask2]).astype(np.float64)

            remaining -= cur_batch
            batch_counter += 1
            if print_progress:
                print(f"[GPU Epsilon Greedy] Completed batch of {cur_batch} episodes. Remaining: {remaining}")

            # logowanie co `log_every` batchy
            if batch_counter % log_every == 0:
                history = log_training_step(history, batch_counter, self, optimal_hard, optimal_soft,
                                            snapshot_policy=True, verbose=False)

            # # opcjonalnie: zapis historii co np. 50 batchy
            # if batch_counter % 50 == 0:
            #     try:
            #         with open(f"history_gpu_es_batch_{batch_counter}.pkl", "wb") as f:
            #             pickle.dump(history, f)
            #     except Exception:
            #         pass
        self.history = history
        return history


    def get_greedy_action(self, state):
        player_sum, dealer_card, usable_ace = state
        dealer_card = map_card_value_host(dealer_card)
        if dealer_card == 11:
            dealer_card = 1
        elif dealer_card > 10:
            dealer_card = 10
        return int(np.argmax(self.Q[player_sum, dealer_card, usable_ace]))
    def evaluate_policy(self, print_n=False):
        print_policy(self.Q, getattr(self, 'counts', None), title="GPU Policy", print_n=print_n)
