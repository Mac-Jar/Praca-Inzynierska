from Blackjack import Card
from Blackjack import Blackjack
import random

class AI:
    def __init__(self):
        self.probabilities_to_hit_table=[]
        self.blackjackGame = Blackjack()
        self.initialize_probabilities_table()

    def initialize_probabilities_table(self):
        self.blackjackGame=Blackjack()
        self.blackjackGame.initialize()
        self.probabilities_to_hit_table=[]
        for player_card1 in self.blackjackGame.deck:
            for player_card2 in self.blackjackGame.deck:
                for dealer_card1 in self.blackjackGame.deck:
                    value_of_card1=player_card1.value
                    value_of_card2=player_card2.value
                    value_of_card3=dealer_card1.value
                    player_cards_value=[value_of_card1,value_of_card2]
                    #dealer_cards_value=[value_of_card3]
                    probability=0.5
                    #self.probabilities_to_hit_table.append([player_cards_value,value_of_card3, probability,0.0,0]) #[player_cards, dealer_cards, probability, total_reward, count]

                    self.probabilities_to_hit_table.append([player_cards_value, value_of_card3,'h', probability, 0.0, 0])  # [player_cards, dealer_cards,action, probability, total_reward, count]
                    self.probabilities_to_hit_table.append([player_cards_value, value_of_card3,'s', probability, 0.0, 0])  # [player_cards, dealer_cards,action, probability, total_reward, count]

    def get_probability(self,player_cards_value,dealer_card_value,action):
        for row in self.probabilities_to_hit_table:
            if sorted(row[0]) == sorted(player_cards_value) and row[1] == dealer_card_value and row[2] == action:
                #return row[2]
                return row[3]
        return 0
    def update_probability(self,player_cards_value,dealer_card_value,action,reward):
        for row in self.probabilities_to_hit_table:
            if sorted(row[0]) == sorted(player_cards_value) and row[1] == dealer_card_value and row[2] == action:
                row[4]+=reward
                row[5]+=1
                avg_reward = row[4] / row[5]
                row[3] = (avg_reward + 1) / 2

    def simulate_games(self,number_of_games=1000):
        for game in range(number_of_games):
            self.blackjackGame.deal_cards(shuffle=True)
            player_cards=self.blackjackGame.player_cards
            dealer_known_card=self.blackjackGame.dealer_cards[0]
            probability_to_hit=self.get_probability(player_cards,dealer_known_card)
            decision=random.choices(
                population=["h","s"],
                weights=[probability_to_hit,1-probability_to_hit],
                k=1
            )
            if decision=="h":
                self.blackjackGame.player_hits()
            game_result = self.blackjackGame.check_game_result()


if __name__ == "__main__":
    simulator = AI()
    simulator.simulate_games()