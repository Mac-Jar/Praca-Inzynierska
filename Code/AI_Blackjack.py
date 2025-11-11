import numpy as np
import random


from Blackjack import Blackjack

class AI_Blackjack:
    def __init__(self,epsilon=0.1,gamma=1.0,alpha=0.1):
        # Q[player_sum][dealer_card][usable_ace][action]
        # action: 0 = stand, 1 = hit
        #dodac licznik do Q ile razy bylem w stanie
        #Q(s,a):= (Q(s,a) * n(s,a)+Reward_za_ten)/n(s,a)+1
        #n(s,a) += 1
        #Dodac tez aktualizowanie takze stanow posrednich o wynik ostatecznych
        self.Q = np.zeros((32, 11, 2, 2))
        #Q - [suma_kart, karta_krupiera, czy_mamy_playable_asa, akcja]
        self.n = np.zeros((32, 11, 2, 2))
        #self.Q = np.zeros((22, 11, 2))
        #self.n = np.zeros((22, 11, 2))
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.blackjackGame = Blackjack()


    def get_state(self):
        player_sum = self.blackjackGame.evaluate_hand(self.blackjackGame.player_cards)
        if player_sum > 21:
            player_sum = 22
        # dealer_card = self.blackjackGame.dealer_cards[0].blackjack_card_value()
        # if dealer_card > 10:
        #     dealer_card = 1
        dealer_card = self.blackjackGame.dealer_cards[0].blackjack_card_value()
        dealer_card = min(max(1, dealer_card), 10)

        #usable_ace = any(c.value == "Ace" for c in self.blackjackGame.player_cards)
        usable_ace = 0
        if any(c.value == "Ace" for c in self.blackjackGame.player_cards) and player_sum < 22:
            usable_ace = 1
        return player_sum, dealer_card, usable_ace
        #return player_sum, dealer_card

    def choose_action(self, state):
        player_sum, dealer_card, usable_ace = state
        #player_sum, dealer_card = state
        # zawsze hit przy sumie < 12
        # if player_sum < 12:
        #     return 1  # HIT

        # eksploracja / eksploatacja
        if random.random() < self.epsilon:
            return random.choice([0, 1])
        #return int(np.argmax(self.Q[player_sum, dealer_card, usable_ace]))


        #max_val = np.max(self.Q[player_sum, dealer_card])
        max_val = np.max(self.Q[player_sum, dealer_card,usable_ace])
        # znajdź wszystkie indeksy, gdzie tab == max_val
        indices = np.flatnonzero(self.Q[player_sum, dealer_card,usable_ace] == max_val)
        # wybierz losowy z tych indeksów
        return np.random.choice(indices)


    def choose_action_final(self, state):
        player_sum, dealer_card, usable_ace = state
        #player_sum, dealer_card = state
        # zawsze hit przy sumie < 12
        # if player_sum < 12:
        #     return 1  # HIT

        return int(np.argmax(self.Q[player_sum, dealer_card, usable_ace]))
        #return int(np.argmax(self.Q[player_sum, dealer_card]))

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
                action = self.choose_action(state)
                actions.append(action)
                states.append(state)

                player_sum = game.evaluate_hand(game.player_cards)
                if player_sum > 21:
                    break
            else:
                break


        reward=game.check_game_result()

        for state,action in zip(states,actions):
            player_sum, dealer_card,usable_ace = state
            self.Q[player_sum,dealer_card,usable_ace,action]=(self.Q[player_sum,dealer_card,usable_ace,action]*self.n[player_sum,dealer_card,usable_ace,action]+reward)/(self.n[player_sum,dealer_card,usable_ace,action]+1)
            self.n[player_sum,dealer_card,usable_ace,action]+=1
            #self.Q[player_sum, dealer_card, action] = (self.Q[player_sum, dealer_card, action] * self.n[player_sum, dealer_card, action] + reward) / (self.n[player_sum, dealer_card, action] + 1)
            #self.n[player_sum, dealer_card, action] += 1


    def train(self, episodes=50000):
        for i in range(episodes):
            self.play_episode()
            if (i + 1) % 5000 == 0:
                print(f"Trenowanie... {i+1}/{episodes}")

    def evaluate_policy(self,print_n=True):
        print("\n=== Strategia AI po treningu ===")
        for usable in [1, 0]:
            print(f"\nUsable Ace = {usable}")
            for dealer_card in range(1, 11):
                decisions = []
                for player_sum in range(12, 22):  # typowy zakres do analizy w Sutton&Barto
                    action = int(np.argmax(self.Q[player_sum, dealer_card, usable]))
                    decision = "HIT" if action == 1 else "STAND"
                    decisions.append(f"{player_sum}:{decision}")
                print(f"Dealer {dealer_card}: " + ", ".join(decisions))

            print("\n--- STAND (akcja 0) ---")
            for player_sum in range(2, 22):
                    row = [int(self.n[player_sum, dealer_card,usable, 0]) for dealer_card in range(2, 11)]
                    print(f"{player_sum:2d}: " + " ".join(f"{val:6d}" for val in row))

            print("\n--- HIT (akcja 1) ---")
            for player_sum in range(2, 22):
                row = [int(self.n[player_sum, dealer_card,usable, 1]) for dealer_card in range(2, 11)]
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
