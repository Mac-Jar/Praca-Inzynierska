from Blackjack import Card
import Blackjack
class AI:
    def __init__(self):
        self.probabilities_to_hit_table=[]

    def initialize_probabilities_table(self):
        self.blackjackGame=Blackjack()
        self.blackjackGame.initialize()
        self.probabilities_to_hit_table=[]
        for card in self.blackjackGame.deck:
            value_of_card=card.value
            probability=0.5
            self.probabilities_to_hit_table.append([value_of_card, probability])
    

