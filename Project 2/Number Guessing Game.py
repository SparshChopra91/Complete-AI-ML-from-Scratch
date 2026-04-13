import random
difficulty = { "easy" : 10 , "hard" : 5}
print("Welcome to the\n<------ NUMBER GUESSING GAME -------->")
print("INSTRUCTIONS\n1.you can select the difficulty level\n2.you have limited number of trials")
print("3.points will be awarded on the basis of the number of attempts you took")
print("Select the Difficulty level :-> \n1.Easy---> 10 , range(1-50) trials\n2.Hard---> 5 trials , range(1-100)")
valid = False
while(valid == False) :
    input_difficulty = input("enter your difficulty level :--------> ")
    if input_difficulty not in difficulty:
        print("Please enter a valid input :-----x ")
    else:
        valid = True
trials = difficulty[input_difficulty.lower()]
if (trials == 10) :
    to_guess = random.randint(1,50)
else:
    to_guess = random.randint(1,100)
attempts = 0
total_trials = trials
output = 0
for i in range(0,trials) :
    valid = 0
    while(valid != 1):
        user = input("please enter the number you guessed :----> ")
        if int(user) is ValueError:
            print("please enter the valid number")
        else:
            user = int(user)
            valid = 1
    attempts += 1
    if(user == to_guess):
        print("<------------YEY!!! you guessed the number------------>")
        output = 1
        break
    diff = abs(user - to_guess)
    if (diff <= 5) :
        print("you are very close to guessing the number")
    elif(diff <= 10 ) :
        print("you are a little bit far from the answer ")
    else :
        print("you are no where close to the answer ")
    if(user > to_guess):
        print("think Lower!!")
    else:
        print("think higher!!")
    print("your remaing attempts are :--> " + str(total_trials-attempts))

if output:
    print("the total number of attempts you took to solve it are :----> " + str(attempts))
    print("your total score is " + str(((total_trials-attempts)/total_trials)*100))
else:
    print("Sorry , you lost the game , All your trials are exhausted\nBetter Luck next time  ")
    print("the Number was :--> " + str(to_guess))
print("x---------EXITING THE GAME-----------x")
exit()