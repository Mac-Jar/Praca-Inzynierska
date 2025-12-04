# utils.py
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
#  OPTIMAL POLICY for Blackjack (HIT=1 / STAND=0)
# ------------------------------------------------------------
def optimal_action(player_sum, dealer_card, usable_ace):
    # Terminal or invalid
    if player_sum < 4 or player_sum > 21:
        return 0

    # Hard sums (no usable ace)
    if usable_ace == 0:
        if player_sum <= 11:
            return 1
        if 12 <= player_sum <= 16:
            return 1 if dealer_card >= 7 else 0
        if player_sum >= 17:
            return 0

    # Soft sums (usable ace)
    if usable_ace == 1:
        if player_sum <= 17:
            return 1
        if player_sum == 18:
            return 1 if dealer_card in (9, 10) else 0
        if player_sum >= 19:
            return 0

    return 0

#
# # ------------------------------------------------------------
# #  Compare learned Q table with optimal policy
# # ------------------------------------------------------------
# def compare_with_optimal(Q):
#     PS_MAX, DC_MAX, UA_MAX, ACTIONS = Q.shape
#
#     correct = []
#     incorrect = []
#
#     agreement_matrix = np.zeros((PS_MAX, DC_MAX, UA_MAX), dtype=np.float32)
#
#     for ps in range(4, 22):
#         ps_idx = ps if ps <= 31 else 31
#         for dc in range(1, 11):
#             for ua in (0, 1):
#                 q0 = Q[ps_idx, dc, ua, 0]
#                 q1 = Q[ps_idx, dc, ua, 1]
#                 learned = 0 if q0 >= q1 else 1
#
#                 optimal = optimal_action(ps, dc, ua)
#
#                 if learned == optimal:
#                     correct.append((ps, dc, ua))
#                     agreement_matrix[ps_idx, dc, ua] = 1
#                 else:
#                     incorrect.append((ps, dc, ua))
#                     agreement_matrix[ps_idx, dc, ua] = -1
#
#     total = len(correct) + len(incorrect)
#     pct = 100 * len(correct) / total if total > 0 else 0
#
#     return {
#         "correct": correct,
#         "incorrect": incorrect,
#         "agreement_pct": pct,
#         "matrix": agreement_matrix
#     }
#
#
# # ------------------------------------------------------------
# #  Visualization
# # ------------------------------------------------------------
# def plot_error_heatmap(agreement_matrix, title="Policy Convergence"):
#     # we visualize only usable_ace=0 slice (more common)
#     m = agreement_matrix[:, :, 0]
#
#     plt.figure(figsize=(10, 6))
#     plt.imshow(m[4:22, 1:11], cmap="coolwarm", vmin=-1, vmax=1, origin="lower")
#     plt.colorbar(label="Agreement (1=correct, -1=incorrect)")
#     plt.xlabel("Dealer showing (1–10)")
#     plt.ylabel("Player sum (4–21)")
#     plt.title(title)
#     plt.show()
def generate_optimal_tables():
    optimal_hard = np.zeros((22, 11), dtype=int)
    optimal_soft = np.zeros((22, 11), dtype=int)

    for ps in range(4, 22):
        for dc in range(1, 11):
            optimal_hard[ps][dc] = optimal_action(ps, dc, 0)
            optimal_soft[ps][dc] = optimal_action(ps, dc, 1)

    return optimal_hard, optimal_soft

def compare_policy_to_optimal(ai, optimal_hard, optimal_soft):
    agreement_hard = np.zeros((22, 11))
    agreement_soft = np.zeros((22, 11))

    for player in range(4, 22):
        for dealer in range(1, 11):
            # HARD TOTAL
            state=[player, dealer, 0]
           #model_action = ai.get_greedy_action(player, dealer, usable_ace=False)
            model_action = ai.get_greedy_action(state)
            agreement_hard[player, dealer] = 1 if model_action == optimal_hard[player][dealer] else -1

            # SOFT TOTAL (usable ace)
            state=[player, dealer, 1 ]

            model_action = ai.get_greedy_action(state)
            agreement_soft[player, dealer] = 1 if model_action == optimal_soft[player][dealer] else -1

    return agreement_hard, agreement_soft

def plot_convergence(agreement, title):
    plt.figure(figsize=(6, 8))
    plt.imshow(agreement[4:22, 1:11], cmap="RdBu", vmin=-1, vmax=1)
    plt.colorbar(label="Agreement (1 = correct, -1 = incorrect)")
    plt.xlabel("Dealer showing (1–10)")
    plt.ylabel("Player sum (4–21)")
    plt.title(title)
    plt.show()
def plot_convergence_dual(agreement_hard, agreement_soft, title="Policy Convergence"):
    fig, axes = plt.subplots(1, 2, constrained_layout=True)

    # HARD TOTALS (no usable ace)
    im0 = axes[0].imshow(
        agreement_hard[4:22, 1:11], cmap="RdBu", vmin=-1, vmax=1
    )
    axes[0].set_title("Hard totals (usable ace = 0)")
    axes[0].set_xlabel("Dealer showing (1–10)")
    axes[0].set_ylabel("Player sum (4–21)")

    # SOFT TOTALS (usable ace = 1)
    im1 = axes[1].imshow(
        agreement_soft[4:22, 1:11], cmap="RdBu", vmin=-1, vmax=1
    )
    axes[1].set_title("Soft totals (usable ace = 1)")
    axes[1].set_xlabel("Dealer showing (1–10)")
    axes[1].set_ylabel("Player sum (4–21)")

    # Wspólna belka kolorów
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Agreement (1 = correct, -1 = incorrect)")

    plt.suptitle(title, fontsize=16)
    plt.show()

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
