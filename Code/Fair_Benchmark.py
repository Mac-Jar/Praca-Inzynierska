import time
import numpy as np
from AI_Blackjack_CPU import *
from AI_Blackjack_CUDA_numba import *
from AI_Blackjack_CUDA_cupy_simplified_blackjack import AI_Blackjack_CuPy
from utils import *
from utils import generate_optimal_tables

# Zakładam, że masz gotowe klasy:
# AI_Blackjack_CPU
# AI_Blackjack_GPU (Numba CUDA)
# AI_Blackjack_CuPy

EPISODES = 10**10
BATCH_SIZE = 100000
optimal_hard, optimal_soft = generate_optimal_tables()
plot_optimal_policy_annotated(optimal_hard, optimal_soft, title="Optimal Strategy")

def benchmark_and_show_policy(ai_class, backend_name="CPU"):
    print(f"=== Benchmark {backend_name} ===")
    ai = ai_class(epsilon=0.1)
    start = time.time()
    if backend_name == "CPU":
        ai.train(episodes=EPISODES)
    else:
        ai.train(episodes=EPISODES)
    end = time.time()
    print(f"{backend_name} training time: {end - start:.2f} s\n")
    # plot_dealer_ace_visits(ai.counts, usable=0)
    # plot_dealer_ace_visits(ai.counts, usable=1)
    ai.evaluate_policy(print_n=False)

    agreement_hard, agreement_soft = compare_policy_to_optimal(
        ai, optimal_hard, optimal_soft
    )

    hard_pct = np.mean(agreement_hard[agreement_hard != 0] == 1) * 100
    soft_mask = np.zeros_like(agreement_soft, dtype=bool)
    soft_mask[12:22, 1:11] = True
    soft_pct = np.mean(agreement_soft[soft_mask] == 1) * 100

    print(f"Hard accuracy: {hard_pct}%")
    print(f"Soft accuracy: {soft_pct:}%")

    # plot_convergence(agreement_hard, title=f"{backend_name} – HARD totals")
    # plot_convergence(agreement_soft, title=f"{backend_name} – SOFT totals")
    plot_convergence_dual(agreement_hard, agreement_soft, title=f"{backend_name} – Convergence")
    plot_learned_policy_annotated(ai, title=f"{backend_name} – Learned Strategy")
    return ai
def benchmark_and_show_policy_es(ai_class, backend_name="CPU ES"):
    print(f"=== Benchmark {backend_name} ===")
    ai = ai_class()
    start = time.time()
    if backend_name == "CPU ES":
        ai.train(episodes=EPISODES)
    else:
        ai.train(episodes=EPISODES)
    end = time.time()

    print(f"{backend_name} training time: {end - start:.2f} s\n")
    # plot_dealer_ace_visits(ai.counts, usable=0)
    # plot_dealer_ace_visits(ai.counts, usable=1)
    ai.evaluate_policy(print_n=False)

    # NOWE porównanie
    agreement_hard, agreement_soft = compare_policy_to_optimal(
        ai, optimal_hard, optimal_soft
    )

    hard_pct = np.mean(agreement_hard[agreement_hard != 0] == 1) * 100
    soft_mask = np.zeros_like(agreement_soft, dtype=bool)
    soft_mask[12:22, 1:11] = True
    soft_pct = np.mean(agreement_soft[soft_mask] == 1) * 100

    print(f"Hard accuracy: {hard_pct:.2f}%")
    print(f"Soft accuracy: {soft_pct:.2f}%")

    # plot_convergence(agreement_hard, title=f"{backend_name} – HARD totals")
    # plot_convergence(agreement_soft, title=f"{backend_name} – SOFT totals")
    plot_convergence_dual(agreement_hard, agreement_soft, title=f"{backend_name} – Convergence")
    plot_learned_policy_annotated(ai, title=f"{backend_name} – Learned Strategy")
    return ai
if __name__ == "__main__":
    # CPU
    #ai_cpu = benchmark_and_show_policy(AI_Blackjack_CPU, "CPU")
    #ai_cpu_es = benchmark_and_show_policy_es(AI_Blackjack_CPU_ES, "CPU ES")

    # GPU Numba
    ai_gpu = benchmark_and_show_policy(AI_Blackjack_GPU, "GPU (Numba CUDA)")
    ai_gpu_es = benchmark_and_show_policy_es(AI_Blackjack_GPU_ES, "GPU ES (Numba CUDA)")

    # GPU CuPy
    #ai_cupy = benchmark_and_show_policy(AI_Blackjack_CuPy, "GPU (CuPy)")

