import random
def value_game(num):
    if num == -1:
        return "Rock" 
    elif num == 0:
        return "Paper"
    else:
        return "Scissors"
print("Welcome to the Rock , Paper and Scissors game :----------> ")
print("the key binds for the game are :-> \nRock\nPaper\nScissors")
user1 = input("please enter your choice :----> ")
game_dict = {"rock" : -1,"paper" : 0,"scissors" : 1}
user = user1.lower()
if user not in game_dict:
    print("enter a valid input please \nExiting")
    exit()
list1 = [-1,0,1]
choice = random.choice(list1)
print("Your Choice : " + str(user) + "\n" + "Computer Choice : " + value_game(choice) )
if(game_dict[user] == choice):
    print("Its a Tie")
elif(game_dict[user]-choice is 1 or -2 ) :
    print("Congratulations you won the game ")
else:
    print("sorry better luck next time")
