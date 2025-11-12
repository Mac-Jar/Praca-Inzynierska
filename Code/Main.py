from AI_Blackjack import AI_Blackjack
from AI_Blackjack_CUDA import AI_Blackjack_CUDA
import time

if __name__ == "__main__":
    episodes = 1000000


    # ai = AI_Blackjack()
    # start = time.time()
    # ai.train(episodes=episodes)
    # end = time.time()
    # ai.evaluate_policy()
    # print(f"Czas wykonania: {end - start:.4f} sekundy")
    ai = AI_Blackjack_CUDA()
    start = time.time()
    ai.train(episodes=episodes)
    end = time.time()
    ai.evaluate_policy()
    print(f"Czas wykonania: {end - start:.4f} sekundy")
