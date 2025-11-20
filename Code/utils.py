# utils.py
import numpy as np

def print_policy(Q, n=None, title="Policy", print_n=False):
    """
    Q: numpy array shape (32,11,2,2) lub cupy array
    n: opcjonalnie liczby odwiedzin w tym samym kształcie
    print_n: czy wypisywać liczbę odwiedzin
    """
    try:
        import cupy as cp
        is_cupy = isinstance(Q, cp.ndarray)
        if is_cupy:
            Q = cp.asnumpy(Q)
        if n is not None and is_cupy:
            n = cp.asnumpy(n)
    except ImportError:
        pass  # tylko numpy

    print(f"\n=== {title} ===")
    for usable in [1, 0]:
        print(f"\nUsable Ace = {usable}")
        for dealer_card in range(1, 11):
            decisions = []
            for player_sum in range(12, 22):
                action = int(np.argmax(Q[player_sum, dealer_card, usable]))
                decisions.append(f"{player_sum}:{('HIT' if action==1 else 'STAND')}")
            print(f"Dealer {dealer_card}: " + ", ".join(decisions))
    if print_n and n is not None:
        print("\n--- Visit counts (sample) ---")
        total_visits = n.sum(axis=(1,2,3)).astype(int)
        for i, val in enumerate(total_visits[:30]):
            print(f"Player sum {i:2d}: {val:6d}")
