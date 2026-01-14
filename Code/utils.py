# utils.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from datetime import datetime

import matplotlib as mpl

mpl.rcParams.update({
    "figure.figsize": (14, 8),      # DOMYŚLNY rozmiar wszystkich wykresów
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

# ------------------------------------------------------------
#  OPTIMAL POLICY for Blackjack (HIT=1 / STAND=0)
# ------------------------------------------------------------

def auto_save_plot(algorithm_name, plot_type):
    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp=""
    safe_name = algorithm_name.replace(" ", "_")
    folder = os.path.join("plots", safe_name)
    os.makedirs(folder, exist_ok=True)
    filename = f"{timestamp}{plot_type}.png"
    return os.path.join(folder, filename)



def save_plot(save_path):
    """
    Zapisuje aktualny wykres do pliku, tworząc katalogi jeśli trzeba.
    Jeśli save_path=None — nic nie zapisuje.
    """
    if save_path is None:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')


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

def plot_learned_policy(ai, title="Learned Policy",save_path=None):
    import matplotlib.pyplot as plt, numpy as np
    Q = ai.Q


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

    if save_plot is not None:
        save_plot(save_path)
    plt.show()
def plot_optimal_policy(optimal_hard, optimal_soft, title="Optimal Policy",save_path=None):
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

    if save_plot is not None:
        save_plot(save_path)

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

def plot_optimal_policy_annotated(optimal_hard, optimal_soft, title="Optimal Strategy",save_path=None):
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
    if save_plot is not None:
        save_plot(save_path)
    plt.show()
def plot_learned_policy_annotated(ai, title="Learned Strategy",save_path=None):
    Q = ai.Q


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
    if save_plot is not None:
        save_plot(save_path)
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
def plot_convergence(agreement, title,save_path=None):
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
    if save_plot is not None:
        save_plot(save_path)
    plt.show()

def compute_accuracy_vs_optimal(ai, optimal_hard, optimal_soft, verbose=True):
    """
    Porównuje politykę agenta `ai` z optymalnymi tabelami (HIT=1 / STAND=0).
    Zwraca dict z metrykami:
      - hard_perstate_pct, soft_perstate_pct, overall_perstate_pct
      - hard_weighted_pct, soft_weighted_pct, overall_weighted_pct
      - counts: odwiedziny per-state (shape (PS_MAX, DC_MAX, UA_MAX))
      - agreement_hard, agreement_soft (macierze -1/0/1)
    Wymaga, by `ai` miał metodę get_greedy_action(state) i atrybut counts (opcjonalnie).
    """
    # 1) porównanie per-state
    agreement_hard, agreement_soft = compare_policy_to_optimal(ai, optimal_hard, optimal_soft)

    # Hard per-state (ignorujemy pola z 0)
    hard_mask = (agreement_hard != 0)
    hard_correct = int((agreement_hard[hard_mask] == 1).sum())
    hard_total = int(hard_mask.sum())
    hard_perstate_pct = 100.0 * hard_correct / hard_total if hard_total > 0 else 0.0

    # Soft per-state: tylko player_sum 12..21, dealer 1..10
    soft_mask_valid = np.zeros_like(agreement_soft, dtype=bool)
    soft_mask_valid[12:22, 1:11] = True
    soft_mask = soft_mask_valid & (agreement_soft != 0)
    soft_correct = int((agreement_soft[soft_mask] == 1).sum())
    soft_total = int(soft_mask.sum())
    soft_perstate_pct = 100.0 * soft_correct / soft_total if soft_total > 0 else 0.0

    # Overall per-state
    overall_correct = hard_correct + soft_correct
    overall_total = hard_total + soft_total
    overall_perstate_pct = 100.0 * overall_correct / overall_total if overall_total > 0 else 0.0

    # 2) visit-weighted accuracy (jeśli dostępne counts)
    counts = getattr(ai, 'counts', None)
    hard_weighted_pct = soft_weighted_pct = overall_weighted_pct = None
    if counts is not None:
        # visits per state (sum over actions)
        visits = counts.sum(axis=-1)  # shape (PS_MAX, DC_MAX, UA_MAX)
        # Hard: slice 4..21, dealer 1..10, usable=0
        hard_visits = visits[4:22, 1:11, 0]
        hard_agree = agreement_hard[4:22, 1:11]  # -1/0/1
        hard_visits_mask = hard_agree != 0
        hard_weighted_correct = float((hard_visits * (hard_agree == 1)).sum())
        hard_weighted_total = float(hard_visits.sum())
        hard_weighted_pct = 100.0 * hard_weighted_correct / hard_weighted_total if hard_weighted_total > 0 else 0.0

        # Soft: usable=1, mask out rows <12
        soft_visits = visits[4:22, 1:11, 1]  # shape (18, 10)
        soft_agree = agreement_soft[4:22, 1:11]  # shape (18, 10)

        # row_mask: True dla wierszy które chcemy zachować (player_sum >= 12)
        row_mask = np.array([ps >= 12 for ps in range(4, 22)], dtype=bool)  # shape (18,)

        # Zastosuj maskę do wierszy (ustaw wiersze <12 na 0)
        soft_agree_masked = soft_agree.copy()
        soft_agree_masked[~row_mask, :] = 0

        # Zastosuj maskę do visits (wyzeruj wiersze <12)
        soft_visits_masked = soft_visits.copy()
        soft_visits_masked[~row_mask, :] = 0

        soft_weighted_correct = float((soft_visits_masked * (soft_agree_masked == 1)).sum())
        soft_weighted_total = float(soft_visits_masked.sum())
        soft_weighted_pct = 100.0 * soft_weighted_correct / soft_weighted_total if soft_weighted_total > 0 else 0.0

        overall_weighted_correct = hard_weighted_correct + soft_weighted_correct
        overall_weighted_total = hard_weighted_total + soft_weighted_total
        overall_weighted_pct = 100.0 * overall_weighted_correct / overall_weighted_total if overall_weighted_total > 0 else 0.0

        hard_weighted_pct = hard_weighted_pct
        soft_weighted_pct = soft_weighted_pct
        overall_weighted_pct = overall_weighted_pct

    # 3) przygotowanie wyniku
    results = {
        "hard_perstate_pct": hard_perstate_pct,
        "soft_perstate_pct": soft_perstate_pct,
        "overall_perstate_pct": overall_perstate_pct,
        "hard_perstate_counts": (hard_correct, hard_total),
        "soft_perstate_counts": (soft_correct, soft_total),
        "agreement_hard": agreement_hard,
        "agreement_soft": agreement_soft,
        "counts": counts,
        "hard_weighted_pct": hard_weighted_pct,
        "soft_weighted_pct": soft_weighted_pct,
        "overall_weighted_pct": overall_weighted_pct
    }

    if verbose:
        print(f"Hard accuracy (per-state): {hard_perstate_pct:.2f}%  ({hard_correct}/{hard_total})")
        print(f"Soft accuracy (per-state): {soft_perstate_pct:.2f}%  ({soft_correct}/{soft_total})")
        print(f"Overall accuracy (per-state): {overall_perstate_pct:.2f}%  ({overall_correct}/{overall_total})")
        if counts is not None:
            print(f"Hard accuracy (visit-weighted): {hard_weighted_pct:.2f}%")
            print(f"Soft accuracy (visit-weighted): {soft_weighted_pct:.2f}%")
            print(f"Overall accuracy (visit-weighted): {overall_weighted_pct:.2f}%")

    return results

# utils.py (dopisz)
import numpy as np

def make_training_history():
    """Zwraca pustą strukturę do logowania historii treningu."""
    return {
        "batches": [],                # numer batchu / epoka
        "hard_perstate": [],          # per-state %
        "soft_perstate": [],
        "overall_perstate": [],
        "hard_weighted": [],          # visit-weighted %
        "soft_weighted": [],
        "overall_weighted": [],
        "policy_hard_snapshots": [],  # list of arrays shape (22,11) or sliced (4:22,1:11)
        "policy_soft_snapshots": [],
    }
def plot_accuracy_history(history, title="Accuracy over training",save_path=None):
    import matplotlib.pyplot as plt
    batches = history["batches"]
    plt.figure(figsize=(10,5))
    # per-state
    plt.plot(batches, history["overall_perstate"], label="Overall per-state")
    plt.plot(batches, history["hard_perstate"], label="Hard per-state", alpha=0.8)
    plt.plot(batches, history["soft_perstate"], label="Soft per-state",  alpha=0.8)

    plt.xlabel("Batch index")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend(loc='best', fontsize='small')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_plot is not None:
        save_plot(save_path)
    plt.show()

def plot_policy_snapshots(history, n_snapshots=6, usable=0,
                          title_prefix="Policy evolution", save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    snaps = history["policy_hard_snapshots"] if usable == 0 else history["policy_soft_snapshots"]
    total = len(snaps)
    if total == 0:
        print("No snapshots recorded.")
        return

    # wybrane indeksy snapshotów
    idxs = np.linspace(0, total - 1, min(n_snapshots, total), dtype=int)

    # ---- SIATKA 2 RZĘDY ----
    n_plots = len(idxs)
    n_rows = 2
    n_cols = math.ceil(n_plots / n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        constrained_layout=True
    )

    axes = np.atleast_2d(axes)  # gwarancja indeksowania [r, c]

    im = None
    for k, i in enumerate(idxs):
        r = k // n_cols
        c = k % n_cols
        ax = axes[r, c]

        data = snaps[i]
        im = ax.imshow(
            data,
            origin='lower',
            cmap='coolwarm',
            vmin=0, vmax=1,
            extent=[1, 10, 12, 21],
            aspect='auto'
        )
        ax.set_title(f"batch {history['batches'][i]}")
        ax.set_xlabel("Dealer")
        ax.set_ylabel("Player sum")

    # wyłącz puste osie (jeśli liczba wykresów nieparzysta)
    for k in range(n_plots, n_rows * n_cols):
        r = k // n_cols
        c = k % n_cols
        axes[r, c].axis("off")

    # wspólny colorbar
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        shrink=0.8,
        label="0 = STAND, 1 = HIT"
    )

    plt.suptitle(f"{title_prefix} (usable={usable})", fontsize=16)

    if save_plot is not None:
        save_plot(save_path)
    plt.show()


def animate_policy(history, usable=0, interval=500, save_path=None):
    """
    Tworzy animację polityki. interval w ms.
    Jeśli save_path podany i masz ffmpeg/kaleido, zapisze plik (mp4/gif).
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import numpy as np

    snaps = history["policy_hard_snapshots"] if usable==0 else history["policy_soft_snapshots"]
    batches = history["batches"]
    if len(snaps) == 0:
        print("No snapshots to animate.")
        return

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(snaps[0], origin='lower', cmap='coolwarm', vmin=0, vmax=1, extent=[1,10,12,21], aspect='auto')
    ax.set_xlabel("Dealer"); ax.set_ylabel("Player sum")
    title = ax.text(0.5, 1.05, f"batch {batches[0]}", transform=ax.transAxes, ha='center')

    def update(frame):
        im.set_data(snaps[frame])
        title.set_text(f"batch {batches[frame]}")
        return im, title

    ani = animation.FuncAnimation(fig, update, frames=len(snaps), interval=interval, blit=False)
    plt.colorbar(im, ax=ax, label="0=STAND,1=HIT")
    plt.suptitle(f"Policy evolution (usable={usable})")
    plt.tight_layout(rect=[0,0,1,0.95])

    if save_path:
        # save as mp4 or gif depending on extension
        ani.save(save_path, writer='ffmpeg', dpi=150)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()


def log_training_step(history, batch_idx, ai, optimal_hard, optimal_soft, snapshot_policy=True, verbose=False):
    """
    Oblicza accuracy i dopisuje do history. Zwraca zaktualizowane history.
    - batch_idx: numer batchu (int)
    - ai: agent (musi mieć get_greedy_action i counts oraz Q)
    - optimal_hard/soft: tabele optymalne
    - snapshot_policy: czy zapisać macierze polityki (można pominąć, by oszczędzić pamięć)
    """
    res = compute_accuracy_vs_optimal(ai, optimal_hard, optimal_soft, verbose=False)
    history["batches"].append(batch_idx)
    history["hard_perstate"].append(res["hard_perstate_pct"])
    history["soft_perstate"].append(res["soft_perstate_pct"])
    history["overall_perstate"].append(res["overall_perstate_pct"])
    history["hard_weighted"].append(res["hard_weighted_pct"])
    history["soft_weighted"].append(res["soft_weighted_pct"])
    history["overall_weighted"].append(res["overall_weighted_pct"])

    if snapshot_policy:
        # snapshot polityki (0..1) dla player 4..21, dealer 1..10
        Q = ai.Q
        policy_hard = np.argmax(Q[:, :, 0, :], axis=-1)  # shape (PS_MAX, DC_MAX)
        policy_soft = np.argmax(Q[:, :, 1, :], axis=-1)
        # przechowujemy tylko zakres 4:22,1:11 żeby oszczędzić miejsce
        history["policy_hard_snapshots"].append(policy_hard[4:22, 1:11].copy())
        history["policy_soft_snapshots"].append(policy_soft[4:22, 1:11].copy())

    if verbose:
        print(f"[log] batch {batch_idx}: overall_perstate={res['overall_perstate_pct']:.2f}%, overall_weighted={res['overall_weighted_pct']:.2f}%")
    return history



def plot_value_3d_matplotlib(ai, title="State-value v* (Matplotlib 3D)",
                             y_base=10, view=(30, 120), box_aspect=(1, 1, 0.5),save_path=None):
    """
    Rysuje 3D surface dla player_sum 12..21 i dealer 1..10.
    Domyślnie ustawia 12 bliżej widza (invert_yaxis=True) i y_base=10/11
    by "linia osi" była na poziomie ~10/11 zamiast 5.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    Q = ai.Q

    # dane: player 12..21, dealer 1..10
    ps_idx = np.arange(12, 22)   # player 12..21
    dc_idx = np.arange(1, 11)    # dealer 1..10
    X, Y = np.meshgrid(dc_idx, ps_idx)  # X: dealer, Y: player

    v_hard = np.max(Q[12:22, 1:11, 0, :], axis=-1)
    v_soft = np.max(Q[12:22, 1:11, 1, :], axis=-1)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, v_soft, cmap='viridis', edgecolor='none')
    ax1.set_title("v* usable ace = 1")
    ax1.set_xlabel("Dealer (1–10)")
    ax1.set_ylabel("Player sum")
    ax1.set_zlabel("Value")
    ax1.view_init(elev=view[0], azim=view[1])
    fig.colorbar(surf1, ax=ax1, shrink=0.6)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, v_hard, cmap='viridis', edgecolor='none')
    ax2.set_title("v* usable ace = 0")
    ax2.set_xlabel("Dealer (1–10)")
    ax2.set_ylabel("Player sum")
    ax2.set_zlabel("Value")
    ax2.view_init(elev=view[0], azim=view[1])
    fig.colorbar(surf2, ax=ax2, shrink=0.6)

    # ustaw dolną granicę osi Y tak, by "linia" była na 10/11
    y_min = float(y_base)   # np. 10 lub 11
    y_max = 21.0
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # odwróć oś Y, żeby mniejsze wartości (12) były bliżej widza
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    # dopasuj proporcje pudełka, żeby zmniejszyć pustą przestrzeń w pionie
    try:
        ax1.set_box_aspect(box_aspect)  # (x, y, z)
        ax2.set_box_aspect(box_aspect)
    except Exception:
        pass  # starsze matplotlib mogą nie wspierać set_box_aspect dla 3D

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()  # <<< KLUCZOWE
    if save_plot is not None:
        save_plot(save_path)
    plt.show()

def plot_convergence_dual(agreement_hard, agreement_soft, title="Policy Convergence",save_path=None):
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
    if save_plot is not None:
        save_plot(save_path)
    plt.show()
def plot_dealer_ace_visits(counts, usable=0,save_path=None):
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
    if save_plot is not None:
        save_plot(save_path)
    plt.show()
def print_policy(Q, n=None, title="Policy", print_n=False):
    """
    Q: numpy array shape (32,11,2,2) lub cupy array
    n: opcjonalnie liczby odwiedzin w tym samym kształcie
    print_n: czy wypisywać liczbę odwiedzin
    """


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
