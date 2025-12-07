# utils.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

def plot_learned_policy(ai, title="Learned Policy"):
    import matplotlib.pyplot as plt, numpy as np
    Q = ai.Q
    try:
        import cupy as cp
        if isinstance(Q, cp.ndarray):
            Q = cp.asnumpy(Q)
    except ImportError:
        pass

    policy_hard = np.zeros((22, 11), dtype=int)
    policy_soft = np.zeros((22, 11), dtype=int)
    for ps in range(4, 22):
        for dc in range(1, 11):
            policy_hard[ps, dc] = int(np.argmax(Q[ps, dc, 0]))
            policy_soft[ps, dc] = int(np.argmax(Q[ps, dc, 1]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    im0 = axes[0].imshow(
        policy_hard[4:22, 1:11],
        cmap="coolwarm",
        vmin=0, vmax=1,
        origin="lower",
        extent=[1, 10, 4, 21]
    )
    axes[0].set_title("Hard totals (usable ace = 0)")
    axes[0].set_xlabel("Dealer showing (1–10)")
    axes[0].set_ylabel("Player sum (4–21)")
    axes[0].set_xticks(range(1, 11))
    axes[0].set_yticks(range(4, 22))

    im1 = axes[1].imshow(
        policy_soft[4:22, 1:11],
        cmap="coolwarm",
        vmin=0, vmax=1,
        origin="lower",
        extent=[1, 10, 4, 21]
    )
    axes[1].set_title("Soft totals (usable ace = 1)")
    axes[1].set_xlabel("Dealer showing (1–10)")
    axes[1].set_ylabel("Player sum (4–21)")
    axes[1].set_xticks(range(1, 11))
    axes[1].set_yticks(range(4, 22))

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Decision (0 = STAND, 1 = HIT)")
    plt.suptitle(title, fontsize=16)
    plt.show()
def plot_optimal_policy(optimal_hard, optimal_soft, title="Optimal Policy"):
    """
    Wizualizuje optymalną strategię HIT/STAND jako macierze
    oddzielnie dla hard totals (usable_ace=0) i soft totals (usable_ace=1).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    im0 = axes[0].imshow(optimal_hard[4:22, 1:11], cmap="coolwarm", vmin=0, vmax=1, origin="lower")
    axes[0].set_title("Hard totals (usable ace = 0)")
    axes[0].set_xlabel("Dealer showing (1–10)")
    axes[0].set_ylabel("Player sum (4–21)")
    axes[0].set_xticks(range(0,10))
    axes[0].set_xticklabels(range(1,11))
    axes[0].set_yticks(range(0,18))
    axes[0].set_yticklabels(range(4,22))

    im1 = axes[1].imshow(optimal_soft[4:22, 1:11], cmap="coolwarm", vmin=0, vmax=1, origin="lower")
    axes[1].set_title("Soft totals (usable ace = 1)")
    axes[1].set_xlabel("Dealer showing (1–10)")
    axes[1].set_ylabel("Player sum (4–21)")
    axes[1].set_xticks(range(0,10))
    axes[1].set_xticklabels(range(1,11))
    axes[1].set_yticks(range(0,18))
    axes[1].set_yticklabels(range(4,22))

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Decision (0 = STAND, 1 = HIT)")

    plt.suptitle(title, fontsize=16)
    plt.show()

def _annotate_cells(ax, data, xvals, yvals, mask=None, hit_char='H', stand_char='S'):
    # data is shape (len(yvals), len(xvals)) with values 0/1
    for yi, y in enumerate(yvals):
        for xi, x in enumerate(xvals):
            if mask is not None and not mask[yi, xi]:
                continue
            v = data[yi, xi]
            txt = hit_char if v == 1 else stand_char
            ax.text(x, y, txt, ha='center', va='center', fontsize=10, color='black')

def plot_optimal_policy_annotated(optimal_hard, optimal_soft, title="Optimal Strategy"):
    # Slice to valid blackjack ranges
    hard = optimal_hard[4:22, 1:11]
    soft = optimal_soft[4:22, 1:11]

    # Axes values
    dealer_vals = list(range(1, 11))   # X
    player_vals = list(range(4, 22))   # Y

    # Mask soft totals below 12 (invalid states)
    soft_mask = np.ones_like(soft, dtype=bool)
    soft_mask[:(12-4), :] = False  # rows 4..11 -> mask off

    # Colormap: 0=Stand (blue), 1=Hit (red), masked=light gray
    cmap = ListedColormap(["#4C78A8", "#E45756"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Hard
    im0 = axes[0].imshow(
        hard, cmap=cmap, vmin=0, vmax=1, origin="lower",
        extent=[1, 10, 4, 21], aspect='auto'
    )
    axes[0].set_title("Hard totals (usable ace = 0)")
    axes[0].set_xlabel("Dealer showing (1–10)")
    axes[0].set_ylabel("Player sum (4–21)")
    axes[0].set_xticks(dealer_vals)
    axes[0].set_yticks(player_vals)
    axes[0].grid(color='white', alpha=0.35, linewidth=0.5)
    _annotate_cells(axes[0], hard, dealer_vals, player_vals)

    # Soft
    soft_plot = soft.copy().astype(float)
    soft_plot[~soft_mask] = np.nan  # mask invalid
    cmap_soft = cmap.copy()
    cmap_soft.set_bad(color="#DDDDDD")
    im1 = axes[1].imshow(
        soft_plot, cmap=cmap_soft, vmin=0, vmax=1, origin="lower",
        extent=[1, 10, 4, 21], aspect='auto'
    )
    axes[1].set_title("Soft totals (usable ace = 1)")
    axes[1].set_xlabel("Dealer showing (1–10)")
    axes[1].set_ylabel("Player sum (4–21)")
    axes[1].set_xticks(dealer_vals)
    axes[1].set_yticks(player_vals)
    axes[1].grid(color='white', alpha=0.35, linewidth=0.5)
    _annotate_cells(axes[1], soft, dealer_vals, player_vals, mask=soft_mask)

    # Small legend (instead of colorbar)
    fig.legend(
        handles=[
            plt.Line2D([], [], marker='s', linestyle='None', color="#E45756", label='HIT'),
            plt.Line2D([], [], marker='s', linestyle='None', color="#4C78A8", label='STAND'),
            plt.Line2D([], [], marker='s', linestyle='None', color="#DDDDDD", label='N/A (soft < 12)'),
        ],
        loc='lower center', ncol=3, frameon=False)

    plt.suptitle(title, fontsize=16)
    plt.show()
def plot_learned_policy_annotated(ai, title="Learned Strategy"):
    Q = ai.Q
    try:
        import cupy as cp
        if isinstance(Q, cp.ndarray):
            Q = cp.asnumpy(Q)
    except ImportError:
        pass

    # Build decision matrices: 0=STAND, 1=HIT
    policy_hard = np.zeros((22, 11), dtype=int)
    policy_soft = np.zeros((22, 11), dtype=int)
    for ps in range(4, 22):
        for dc in range(1, 11):
            policy_hard[ps, dc] = int(np.argmax(Q[ps, dc, 0]))
            policy_soft[ps, dc] = int(np.argmax(Q[ps, dc, 1]))

    hard = policy_hard[4:22, 1:11]
    soft = policy_soft[4:22, 1:11]

    dealer_vals = list(range(1, 11))
    player_vals = list(range(4, 22))

    soft_mask = np.ones_like(soft, dtype=bool)
    soft_mask[:(12-4), :] = False

    cmap = ListedColormap(["#4C78A8", "#E45756"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Hard
    axes[0].imshow(
        hard, cmap=cmap, vmin=0, vmax=1, origin="lower",
        extent=[1, 10, 4, 21], aspect='auto'
    )
    axes[0].set_title("Hard totals (usable ace = 0)")
    axes[0].set_xlabel("Dealer showing (1–10)")
    axes[0].set_ylabel("Player sum (4–21)")
    axes[0].set_xticks(dealer_vals)
    axes[0].set_yticks(player_vals)
    axes[0].grid(color='white', alpha=0.35, linewidth=0.5)
    _annotate_cells(axes[0], hard, dealer_vals, player_vals)

    # Soft
    soft_plot = soft.copy().astype(float)
    soft_plot[~soft_mask] = np.nan
    cmap_soft = cmap.copy()
    cmap_soft.set_bad(color="#DDDDDD")
    axes[1].imshow(
        soft_plot, cmap=cmap_soft, vmin=0, vmax=1, origin="lower",
        extent=[1, 10, 4, 21], aspect='auto'
    )
    axes[1].set_title("Soft totals (usable ace = 1)")
    axes[1].set_xlabel("Dealer showing (1–10)")
    axes[1].set_ylabel("Player sum (4–21)")
    axes[1].set_xticks(dealer_vals)
    axes[1].set_yticks(player_vals)
    axes[1].grid(color='white', alpha=0.35, linewidth=0.5)
    _annotate_cells(axes[1], soft, dealer_vals, player_vals, mask=soft_mask)

    fig.legend(
        handles=[
            plt.Line2D([], [], marker='s', linestyle='None', color="#E45756", label='HIT'),
            plt.Line2D([], [], marker='s', linestyle='None', color="#4C78A8", label='STAND'),
            plt.Line2D([], [], marker='s', linestyle='None', color="#DDDDDD", label='N/A (soft < 12)'),
        ],
        loc='lower center', ncol=3, frameon=False    )

    plt.suptitle(title, fontsize=16)
    plt.show()
def true_table_optimal_action(player_sum, dealer_card, usable_ace):
    # Terminal or invalid
    if player_sum < 4 or player_sum > 21:
        return 0  # STAND (nieważne)

    # Hard totals (usable_ace = 0)
    if usable_ace == 0:
        # Always HIT ≤ 11
        if player_sum <= 11:
            return 1
        # Hard 12: HIT vs 2–3 and 7–A; STAND vs 4–6
        if player_sum == 12:
            if dealer_card in (4, 5, 6):
                return 0  # STAND
            else:
                return 1  # HIT
        # Hard 13–16: HIT vs 7–A; STAND vs 2–6
        if 13 <= player_sum <= 16:
            return 1 if dealer_card >= 7 else 0
        # Hard ≥17: STAND
        if player_sum >= 17:
            return 0

    # Soft totals (usable_ace = 1)
    if usable_ace == 1:
        # Soft ≤17: always HIT
        if player_sum <= 17:
            return 1
        # Soft 18: HIT vs 9–10; STAND otherwise
        if player_sum == 18:
            return 1 if dealer_card in (9, 10) else 0
        # Soft ≥19: STAND
        if player_sum >= 19:
            return 0

    # Fallback
    return 0
def generate_optimal_tables():
    optimal_hard = np.zeros((22, 11), dtype=int)
    optimal_soft = np.zeros((22, 11), dtype=int)
    for ps in range(4, 22):
        for dc in range(1, 11):
            optimal_hard[ps][dc] = true_table_optimal_action(ps, dc, 0)
            optimal_soft[ps][dc] = true_table_optimal_action(ps, dc, 1)
    return optimal_hard, optimal_soft
def compare_policy_to_optimal(ai, optimal_hard, optimal_soft):
    agreement_hard = np.zeros((22, 11))
    agreement_soft = np.zeros((22, 11))

    for player in range(4, 22):
        for dealer in range(1, 11):
            # HARD TOTAL
            state = [player, dealer, 0]
            model_action = ai.get_greedy_action(state)
            agreement_hard[player, dealer] = 1 if model_action == optimal_hard[player][dealer] else -1

            # SOFT TOTAL — tylko dla player_sum ≥ 12
            if player >= 12:
                state = [player, dealer, 1]
                model_action = ai.get_greedy_action(state)
                agreement_soft[player, dealer] = 1 if model_action == optimal_soft[player][dealer] else -1

    return agreement_hard, agreement_soft
def plot_convergence(agreement, title):
    import matplotlib.pyplot as plt
    # Slice to valid blackjack ranges: player 4..21, dealer 1..10
    data = agreement[4:22, 1:11]

    plt.figure(figsize=(6, 8))
    im = plt.imshow(
        data,
        cmap="RdBu",
        vmin=-1, vmax=1,
        origin="lower",
        extent=[1, 10, 4, 21]  # X: dealer 1..10, Y: player 4..21
    )
    plt.colorbar(im, label="Agreement (1 = correct, -1 = incorrect)")
    plt.xlabel("Dealer showing (1–10)")
    plt.ylabel("Player sum (4–21)")

    # Set ticks to real blackjack values (not array indices)
    plt.xticks(range(1, 11))
    plt.yticks(range(4, 22))

    plt.title(title)
    plt.show()
# def plot_convergence_dual(agreement_hard, agreement_soft, title="Policy Convergence"):
#     import matplotlib.pyplot as plt
#     # Slice to valid blackjack ranges
#     hard = agreement_hard[4:22, 1:11]
#     soft = agreement_soft[4:22, 1:11]
#
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
#
#     im0 = axes[0].imshow(
#         hard,
#         cmap="RdBu",
#         vmin=-1, vmax=1,
#         origin="lower",
#         extent=[1, 10, 4, 21]
#     )
#     axes[0].set_title("Hard totals (usable ace = 0)")
#     axes[0].set_xlabel("Dealer showing (1–10)")
#     axes[0].set_ylabel("Player sum (4–21)")
#     axes[0].set_xticks(range(1, 11))
#     axes[0].set_yticks(range(4, 22))
#
#     im1 = axes[1].imshow(
#         soft,
#         cmap="RdBu",
#         vmin=-1, vmax=1,
#         origin="lower",
#         extent=[1, 10, 4, 21]
#     )
#     axes[1].set_title("Soft totals (usable ace = 1)")
#     axes[1].set_xlabel("Dealer showing (1–10)")
#     axes[1].set_ylabel("Player sum (4–21)")
#     axes[1].set_xticks(range(1, 11))
#     axes[1].set_yticks(range(4, 22))
#
#     cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)
#     cbar.set_label("Agreement (1 = correct, -1 = incorrect)")
#     plt.suptitle(title, fontsize=16)
#     plt.show()
def plot_convergence_dual(agreement_hard, agreement_soft, title="Policy Convergence"):
    hard = agreement_hard[4:22, 1:11]
    soft = agreement_soft[4:22, 1:11]

    dealer_vals = list(range(1, 11))
    player_vals = list(range(4, 22))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    im0 = axes[0].imshow(hard, cmap="RdBu", vmin=-1, vmax=1, origin="lower",
                         extent=[1, 10, 4, 21], aspect='auto')
    axes[0].set_title("Hard totals (usable ace = 0)")
    axes[0].set_xlabel("Dealer showing (1–10)")
    axes[0].set_ylabel("Player sum (4–21)")
    axes[0].set_xticks(dealer_vals)
    axes[0].set_yticks(player_vals)
    axes[0].grid(color='white', alpha=0.35, linewidth=0.5)

    im1 = axes[1].imshow(soft, cmap="RdBu", vmin=-1, vmax=1, origin="lower",
                         extent=[1, 10, 4, 21], aspect='auto')
    axes[1].set_title("Soft totals (usable ace = 1)")
    axes[1].set_xlabel("Dealer showing (1–10)")
    axes[1].set_ylabel("Player sum (4–21)")
    axes[1].set_xticks(dealer_vals)
    axes[1].set_yticks(player_vals)
    axes[1].grid(color='white', alpha=0.35, linewidth=0.5)

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Agreement (1 = correct, -1 = incorrect)")
    plt.suptitle(title, fontsize=16)
    plt.show()
def plot_dealer_ace_visits(counts, usable=0):
    import matplotlib.pyplot as plt
    # visits dla dealer=Ace (kolumna 1)
    visits = counts[:, 1, usable, :].sum(axis=-1)
    # przycięcie do player_sum 4..21
    ps_range = range(4, 22)
    visits_slice = visits[4:22]

    plt.plot(ps_range, visits_slice, marker='o')
    plt.title(f"Visits vs player_sum (dealer=Ace, usable={usable})")
    plt.xlabel("Player sum")
    plt.ylabel("Visits")
    plt.grid(True)
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
