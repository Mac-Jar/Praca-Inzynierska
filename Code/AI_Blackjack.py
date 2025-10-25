import numpy as np
import random
from Blackjack import Blackjack  # Twój plik z klasą Blackjack

class AI_Blackjack:
    def __init__(self):
        # Q[player_sum][dealer_card][usable_ace][action]
        # action: 0 = stand, 1 = hit
        #dodac licznik do Q ile razy bylem w stanie
        #Q(s,a):= (Q(s,a) * n(s,a)+Reward_za_ten)/n(s,a)+1
        #n(s,a) += 1
        #Dodac tez aktualizowanie takze stanow posrednich o wynik ostatecznych
        #self.Q = np.zeros((32, 11, 2, 2))
        #self.n = np.zeros((32, 11, 2, 2))
        self.Q = np.zeros((22, 11, 2))
        self.n = np.zeros((22, 11, 2))
        self.epsilon = 0.1
        self.gamma = 1.0
        self.alpha = 0.1
        self.blackjackGame = Blackjack()


    def get_state(self):
        player_sum = self.blackjackGame.evaluate_hand(self.blackjackGame.player_cards)
        if player_sum > 21:
            player_sum = 22
        dealer_card = self.blackjackGame.dealer_cards[0].blackjack_card_value()
        if dealer_card > 10:
            dealer_card = 1
        #usable_ace = any(c.value == "Ace" for c in self.blackjackGame.player_cards)
        #return player_sum, dealer_card, int(usable_ace)
        return player_sum, dealer_card

    def choose_action(self, state):
        #player_sum, dealer_card, usable_ace = state
        player_sum, dealer_card = state
        # zawsze hit przy sumie < 12
        # if player_sum < 12:
        #     return 1  # HIT

        # eksploracja / eksploatacja
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        #return int(np.argmax(self.Q[player_sum, dealer_card, usable_ace]))


        max_val = np.max(self.Q[player_sum, dealer_card])
        # znajdź wszystkie indeksy, gdzie tab == max_val
        indices = np.flatnonzero(self.Q[player_sum, dealer_card] == max_val)
        # wybierz losowy z tych indeksów
        return np.random.choice(indices)


    def choose_action_final(self, state):
        #player_sum, dealer_card, usable_ace = state
        player_sum, dealer_card = state
        # zawsze hit przy sumie < 12
        # if player_sum < 12:
        #     return 1  # HIT

        #return int(np.argmax(self.Q[player_sum, dealer_card, usable_ace]))
        return int(np.argmax(self.Q[player_sum, dealer_card]))

    def play_episode(self):
        game = self.blackjackGame
        game.initialize()
        game.deal_cards(shuffle=True)

        state = self.get_state()
        states=[]
        actions=[]
        #done = False
        if game.evaluate_hand(game.player_cards)==4 and game.dealer_cards[0].blackjack_card_value() == 7:
            tmp=1

        action = self.choose_action(state)
        actions.append(action)
        states.append(state)
        while action!=0:

            if action == 1:
                game.player_hits()
                state=self.get_state()
                player_sum = game.evaluate_hand(game.player_cards)
                if player_sum > 21:
                    break

                action = self.choose_action(state)
                actions.append(action)
                states.append(state)
            else:
                break


        reward=game.check_game_result()

        for state,action in zip(states,actions):
            player_sum, dealer_card = state
            # self.Q[player_sum,dealer_card,usable_ace,action]=(self.Q[player_sum,dealer_card,usable_ace,action]*self.n[player_sum,dealer_card,usable_ace,action]+reward)/(self.n[player_sum,dealer_card,usable_ace,action]+1)
            # self.n[player_sum,dealer_card,usable_ace,action]+=1
            self.Q[player_sum, dealer_card, action] = (self.Q[player_sum, dealer_card, action] * self.n[player_sum, dealer_card, action] + reward) / (self.n[player_sum, dealer_card, action] + 1)
            self.n[player_sum, dealer_card, action] += 1


#             if action == 1:  # hit
#                 game.player_hits()
#             else:
#                 done = True
#
#             reward = 0
#             player_sum = game.evaluate_hand(game.player_cards)
#             if player_sum > 21:
#                 reward = -1
#                 done = True
#             elif done:
#                 reward = game.check_game_result()
#
#             next_state = self.get_state() if not done else None
#
# #w blackjacku pomijam nagrody czesciowe
#
#             # aktualizacja Q
#             p_sum, d_card, u_ace = state
#             if next_state:
#                 ns_p, ns_d, ns_u = next_state
#                 best_next = np.max(self.Q[ns_p, ns_d, ns_u])
#                 td_target = reward + self.gamma * best_next
#             else:
#                 td_target = reward
#
#             td_error = td_target - self.Q[p_sum, d_card, u_ace, action]
#             self.Q[p_sum, d_card, u_ace, action] += self.alpha * td_error
#
#             state = next_state if next_state else state

    def train(self, episodes=50000):
        for i in range(episodes):
            self.play_episode()
            if (i + 1) % 5000 == 0:
                print(f"Trenowanie... {i+1}/{episodes}")

    def evaluate_policy(self):
        print("\n=== Strategia AI po treningu ===")
        #print("(Ace=1 oznacza, że w ręce jest as liczący się jako 11)\n")
        for dealer_card in range(2, 11):
            decisions = []
            for player_sum in range(2, 22):
                #action = np.argmax(self.Q[player_sum, dealer_card, usable_ace])
                state = [player_sum, dealer_card]
                action = self.choose_action_final(state)
                decision = "HIT  " if action == 1 else "STAND"
                decisions.append(f"{player_sum}:{decision}")
            print(f"Dealer {dealer_card}: " + ", ".join(decisions))
        print("\n=== Tablica odwiedzin (liczba wystąpień stanu) ===")

        print("\n--- STAND (akcja 0) ---")
        for player_sum in range(2, 22):
            row = [int(self.n[player_sum, dealer_card, 0]) for dealer_card in range(2, 11)]
            print(f"{player_sum:2d}: " + " ".join(f"{val:6d}" for val in row))

        print("\n--- HIT (akcja 1) ---")
        for player_sum in range(2, 22):
            row = [int(self.n[player_sum, dealer_card, 1]) for dealer_card in range(2, 11)]
            print(f"{player_sum:2d}: " + " ".join(f"{val:6d}" for val in row))

        # print("Tablica odwiedzin: dla stand")
        # # for i in range(11):
        # #     for j in range(22):
        # #         print(f"{self.n[i][j][0]} ")
        # #     print()
        # for row in self.n[:, :, 0]:
        #         print(' '.join(map(str, row.astype(int))))
        # print("Tablica odwiedzin: dla hit")
        # # for i in range(11):
        # #     for j in range(22):
        # #         print(f"{self.n[i][j][1]} ")
        # #     print()
        # for row in self.n[:, :, 1]:
        #         print(' '.join(map(str, row.astype(int))))
