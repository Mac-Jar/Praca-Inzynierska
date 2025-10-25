from AI_Blackjack import AI_Blackjack
from cop import AI_Blackjack2

import time

if __name__ == "__main__":
    ai = AI_Blackjack()
    start = time.time()
    ai.train(episodes=100000)
    end = time.time()
    ai.evaluate_policy()
    print(f"Czas wykonania: {end - start:.4f} sekundy")
    ai = AI_Blackjack2()
    start = time.time()
    ai.train(episodes=100000)
    end = time.time()
    ai.evaluate_policy()
    print(f"Czas wykonania: {end - start:.4f} sekundy")

