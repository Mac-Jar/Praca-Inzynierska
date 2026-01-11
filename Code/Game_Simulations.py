# ============================================================
# === SYMULACJA GRY — PORÓWNANIE STRATEGII PO TRENINGU ========
# ============================================================
from utils import generate_optimal_tables
import numpy as np
optimal_hard, optimal_soft = generate_optimal_tables()

def simulate_blackjack_episode(policy_fn):
    """
    policy_fn(state) -> action (0=STAND, 1=HIT)
    Zwraca reward: -1, 0, 1
    """
    # losowanie kart
    def draw():
        v = np.random.randint(2, 15)
        if 11 <= v <= 13:
            return 10
        if v == 14:
            return 11
        return v

    def evaluate(hand):
        total = sum(hand)
        aces = hand.count(11)
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        usable = 1 if aces > 0 else 0
        return total, usable

    # initial
    player = [draw(), draw()]
    dealer = [draw(), draw()]

    # player turn
    while True:
        p_sum, usable = evaluate(player)
        if p_sum > 21:
            return -1  # bust

        dealer_up = dealer[0]
        if dealer_up == 11:
            dealer_up = 1
        elif dealer_up > 10:
            dealer_up = 10

        action = policy_fn((p_sum, dealer_up, usable))

        if action == 1:  # HIT
            player.append(draw())
        else:
            break

    # dealer turn
    while True:
        d_sum, _ = evaluate(dealer)
        if d_sum >= 17:
            break
        dealer.append(draw())

    d_sum, _ = evaluate(dealer)
    p_sum, _ = evaluate(player)

    if d_sum > 21:
        return 1
    if p_sum > d_sum:
        return 1
    if p_sum < d_sum:
        return -1
    return 0


# === STRATEGIE ===

def policy_ai(ai):
    return lambda state: ai.get_greedy_action(state)

def policy_optimal(state):
    p_sum, dealer, usable = state
    if usable == 1:
        return optimal_soft[p_sum][dealer]
    else:
        return optimal_hard[p_sum][dealer]

def policy_random(state):
    return np.random.randint(0, 2)

def policy_human(state):
    p_sum, dealer, usable = state
    if p_sum <= 12:
        return 1
    if 13 <= p_sum <= 16:
        return 1 if dealer >= 7 else 0
    return 0


# === SYMULATOR ZBIORCZY ===

def simulate_many(policy_fn, n):
    results = [simulate_blackjack_episode(policy_fn) for _ in range(n)]
    wins = results.count(1)
    losses = results.count(-1)
    draws = results.count(0)
    return wins, losses, draws, wins / n


def run_simulation(ai, label="GPU AI", n=200000):
    print(f"\n=== Symulacja strategii: {label} ===")

    tests = {
        "AI vs Dealer": policy_ai(ai),
        "Optimal vs Dealer": policy_optimal,
        "Random vs Dealer": policy_random,
        "Human-like vs Dealer": policy_human,
    }

    for name, pol in tests.items():
        w, l, d, wr = simulate_many(pol, n)
        print(f"{name}: W={w}, L={l}, D={d}, WinRate={wr*100:.3f}%")

    print("\n")


# ============================================================
# === WYKONANIE SYMULACJI PO TRENINGU ========================
# ============================================================

