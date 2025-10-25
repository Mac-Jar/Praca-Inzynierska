import random


class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    def __str__(self):
        return f'{self.value} {self.suit}'
    def blackjack_card_value(self):
        if self.value in ["King", "Queen", "Jack"]:
            return 10
        elif self.value in ["Ace"]:
            return 11
        else:
            return int(self.value)
class Blackjack:
    def __init__(self,play_manually=False):
        self.deck = []
        self.player_cards=[]
        self.dealer_cards=[]
        self.player_points = 0
        self.dealer_points = 0
        self.play_manually = play_manually

    def make_deck(self):
        self.deck = []
        suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
        for i in range(2, 11):
            for suit in suits:
                self.deck.append(Card(suit, i))
        for j in range(4):
            self.deck.append(Card(suits[j], "Ace"))
            self.deck.append(Card(suits[j], "King"))
            self.deck.append(Card(suits[j], "Queen"))
            self.deck.append(Card(suits[j], "Jack"))
        self.shuffle_deck()

    def initialize(self):
        self.deck=[]
        suits=["Hearts","Diamonds","Clubs","Spades"]
        for i in range(2,9):
            for j in range(4):
                self.deck.append(Card(suits[j],i))
        for j in range(4):
            self.deck.append(Card(suits[j], "Ace"))
            self.deck.append(Card(suits[j], "King"))
            self.deck.append(Card(suits[j], "Queen"))
            self.deck.append(Card(suits[j], "Jack"))
        self.shuffle_deck()
    def deal_cards(self,shuffle=False):
        self.deck.extend(self.player_cards)
        self.deck.extend(self.dealer_cards)
        self.player_cards = []
        self.dealer_cards = []
        #self.play_manually = play_manually
        if shuffle:
            random.shuffle(self.deck)
        self.player_cards.append(self.deck.pop())
        self.dealer_cards.append(self.deck.pop())
        self.player_cards.append(self.deck.pop())
        self.dealer_cards.append(self.deck.pop())
        # self.used_cards = []
        # self.used_cards.append(self.dealers_cards[0])
        # self.used_cards.append(self.dealers_cards[1])
        # self.used_cards.append(self.player_cards[0])
        # self.used_cards.append(self.player_cards[1])

    def print_cards(self):
        print("Your cards:")
        for card in self.player_cards:
            print(card)
        print(f"Your points: {self.player_points}")
        print("Dealer known card:")
        print(self.dealer_cards[0])
        print(f"Dealer known points: {self.dealer_cards[0].blackjack_card_value()}")

    def print_all_cards(self):
        print("Player cards:")
        for card in self.player_cards:
            print(card)
        print(f"Your points: {self.player_points}")

        print("Dealer cards:")
        for card in self.dealer_cards:
            print(card)
        print(f"Dealer  points: {self.dealer_points}")

    def shuffle_deck(self):
        random.shuffle(self.deck)
    def start_game(self):
        self.initialize()
        self.play_games()
    def player_hits(self):
        self.player_cards.append(self.deck.pop())
    def play_games(self,number_of_games=10):
        i=0
        while True:
            if i>number_of_games:
                break
            i+=1

            if self.play_manually:
                self.deal_cards()
                game_over=False
                while not game_over:
                    self.player_points=self.evaluate_hand(self.player_cards)
                    self.print_cards()
                    if self.player_points>21:
                        break
                    print("Do you want to hit or stand? (h/s)")
                    decision=input()
                    if decision=="h":

                        self.player_cards.append(self.deck.pop())
                        #self.used_cards.append(self.player_cards[-1])
                    elif decision=="s":
                        game_over=True
                    else:
                        print("Wrong input. Try again.")
                game_result=self.check_game_result()
                if game_result==1:
                    print("You won")
                elif game_result==0:
                    print("Game ended in a draw")
                else:
                    print("You lost")

                print("Do you  want to continue playing? (y/n)")
                decision=input()
                if decision=="n":
                    break

    def evaluate_hand(self,hand):
        total_points = 0
        number_of_aces = 0
        for card in hand:
            total_points += card.blackjack_card_value()
            if card.value == "Ace":
                number_of_aces += 1
        while total_points > 21 and number_of_aces >= 1:
            total_points -= 10
            number_of_aces -= 1
        return total_points
    def check_game_result(self,):

        self.player_points = self.evaluate_hand(self.player_cards)
        if self.player_points>21:
            return -1
        self.dealer_points = self.evaluate_hand(self.dealer_cards)
        if self.play_manually:
            self.print_all_cards()

        while self.dealer_points<17:
            if self.play_manually:
                print("Dealer must hit")
                x=input()
            self.dealer_cards.append(self.deck.pop())
            #self.used_cards.append(self.dealers_cards[-1])
            self.dealer_points = self.evaluate_hand(self.dealer_cards)
            if self.play_manually:
                self.print_all_cards()


        if self.dealer_points>21:
            return 1
        if self.dealer_points < self.player_points:
            return 1
        elif self.dealer_points == self.player_points:
            return 0
        else:
            return -1


if __name__ == "__main__":
    game = Blackjack(play_manually=True)
    game.start_game()
