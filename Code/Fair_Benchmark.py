from AI_Blackjack_CPU_EpsilonGreedy import *
from AI_Blackjack_CPU_ES import *
from AI_Blackjack_CUDA_EpsilonGreedy import *
from AI_Blackjack_CUDA_ES import *
from Game_Simulations import *
from utils import *

EPISODES = 10**10
optimal_hard, optimal_soft = generate_optimal_tables()
plot_optimal_policy_annotated(optimal_hard, optimal_soft, title="Optimal Strategy")

def benchmark_and_show_policy(ai_class, algorithm_name="CPU EpsilonGreedy"):
    print(f"=== Benchmark {algorithm_name} ===")
    ai = ai_class(epsilon=0.1)
    start = time.time()

    history=ai.train(episodes=EPISODES)
    end = time.time()
    print(f"{algorithm_name} training time: {end - start:.2f} s\n")

    ai.evaluate_policy(print_n=True)

    agreement_hard, agreement_soft = compare_policy_to_optimal(
        ai, optimal_hard, optimal_soft
    )

    results = compute_accuracy_vs_optimal(ai, optimal_hard, optimal_soft, verbose=False)
    # dostęp do wartości:
    hard_pct = results["hard_perstate_pct"]
    soft_pct = results["soft_perstate_pct"]
    overall_pct = results["overall_perstate_pct"]
    print(f"Hard accuracy: {hard_pct}%")
    print(f"Soft accuracy: {soft_pct:}%")
    print(f"Overall accuracy: {overall_pct:}%")


    #wykresy:
    plot_accuracy_history(
        history,
        title="GPU Eosilon — Accuracy over training",
        save_path=auto_save_plot(algorithm_name, "accuracy")
    )

    plot_policy_snapshots(
        history,
        n_snapshots=6,
        usable=0,
        title_prefix="GPU ES policy (hard)",
        save_path=auto_save_plot(algorithm_name, "policy_hard")
    )

    plot_policy_snapshots(
        history,
        n_snapshots=6,
        usable=1,
        title_prefix="GPU Eosilon policy (soft)",
        save_path=auto_save_plot(algorithm_name, "policy_soft")
    )

    plot_convergence_dual(
        agreement_hard,
        agreement_soft,
        title=f"{algorithm_name} – Convergence",
        save_path=auto_save_plot(algorithm_name, "convergence")
    )

    plot_learned_policy_annotated(
        ai,
        title=f"{algorithm_name} – Learned Strategy",
        save_path=auto_save_plot(algorithm_name, "learned_strategy")
    )

    plot_value_3d_matplotlib(
        ai,
        title=f"{algorithm_name} — Value & Policy",
        save_path=auto_save_plot(algorithm_name, "value_surface")
    )

    print(f"{algorithm_name} training time: {end - start:.2f} s\n")

    return ai
def benchmark_and_show_policy_es(ai_class, algorithm_name="CPU ES"):
    print(f"=== Benchmark {algorithm_name} ===")
    ai = ai_class()
    start = time.time()

    history = ai.train(episodes=EPISODES)
    end = time.time()
    print(f"{algorithm_name} training time: {end - start:.2f} s\n")

    ai.evaluate_policy(print_n=True)

    agreement_hard, agreement_soft = compare_policy_to_optimal(
        ai, optimal_hard, optimal_soft
    )

    results = compute_accuracy_vs_optimal(ai, optimal_hard, optimal_soft, verbose=False)
    # dostęp do wartości:
    hard_pct = results["hard_perstate_pct"]
    soft_pct = results["soft_perstate_pct"]
    overall_pct = results["overall_perstate_pct"]
    print(f"Hard accuracy: {hard_pct}%")
    print(f"Soft accuracy: {soft_pct:}%")
    print(f"Overall accuracy: {overall_pct:}%")

    # wykresy:
    plot_accuracy_history(
        history,
        title="GPU Eosilon — Accuracy over training",
        save_path=auto_save_plot(algorithm_name, "accuracy")
    )

    plot_policy_snapshots(
        history,
        n_snapshots=6,
        usable=0,
        title_prefix="GPU ES policy (hard)",
        save_path=auto_save_plot(algorithm_name, "policy_hard")
    )

    plot_policy_snapshots(
        history,
        n_snapshots=6,
        usable=1,
        title_prefix="GPU Eosilon policy (soft)",
        save_path=auto_save_plot(algorithm_name, "policy_soft")
    )

    plot_convergence_dual(
        agreement_hard,
        agreement_soft,
        title=f"{algorithm_name} – Convergence",
        save_path=auto_save_plot(algorithm_name, "convergence")
    )

    plot_learned_policy_annotated(
        ai,
        title=f"{algorithm_name} – Learned Strategy",
        save_path=auto_save_plot(algorithm_name, "learned_strategy")
    )

    plot_value_3d_matplotlib(
        ai,
        title=f"{algorithm_name} — Value & Policy",
        save_path=auto_save_plot(algorithm_name, "value_surface")
    )

    print(f"{algorithm_name} training time: {end - start:.2f} s\n")
    return ai
if __name__ == "__main__":
    # CPU
    #ai_cpu = benchmark_and_show_policy(AI_Blackjack_CPU, "CPU")
    #ai_cpu_es = benchmark_and_show_policy_es(AI_Blackjack_CPU_ES, "CPU ES")

    # GPU Numba
    ai_gpu = benchmark_and_show_policy(AI_Blackjack_GPU, "GPU EpsilonGreedy")
    ai_gpu_es = benchmark_and_show_policy_es(AI_Blackjack_GPU_ES, "GPU ES")
    number_of_games_to_simulate = 5*10**5
    run_simulation(ai_gpu, "GPU MC-EpsilonGreedy",n=number_of_games_to_simulate)
    run_simulation(ai_gpu_es, "GPU MC-ES",n=number_of_games_to_simulate)
