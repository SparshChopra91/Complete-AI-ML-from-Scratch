"""

#Write a program to read the text from a given file poems.txt and find out 
#whether it contains the word twinkle.

file1 = open("F:\\Learning\\AI ML Learn from Scratch\\DAY 8\\poems.txt" , "r")
print("this program will tell whether the word Twinkle is there in the file or not ")
word = "twinkle"
words = "."
found = False
while (words != ""):
    words = file1.readline()
    if word in words.lower():
        print("yes the word twinkle is in the file ")
        found = True
        break
if(found is False):
    print("the twinkle was not there in the file ")

    
#The game() function in a program lets a user play a game and returns the score 
#as an integer. You need to read a file ‘Hi-score.txt’ which is either blank or 
#contains the previous Hi-score. You need to write a program to update the Hi
#score whenever the game() function breaks the Hi-score.

import random 
def game():
    return random.randint(1,100)
try:
    file = open("F:\\Learning\\AI ML Learn from Scratch\\DAY 8\\Hi-Score.txt" , "r+")
except FileNotFoundError:
    print("file dosen't exist in the computer")
    exit()
new_score = game()
data = file.read()
if(data == ""):
    print("No score found in the file so your new score is all time high that is " + str(new_score))
else:
    old_score = int(data)
    if(new_score > old_score):
        file.seek(0)
        file.truncate()
        file.write(str(new_score))
        print("your new high score " + str(new_score) + " had been updated in the file ")
    else:
        print("your previous high score " + str(old_score) + " is greater than the new score of " + str(new_score))
file.close()


#Write a program to generate multiplication tables from 2 to 20 and write it to the 
#different files. Place these files in a folder for a 13 – year old.

valid = False
while (valid is False):
    num = input("please enter the number of table you want to get in files starting from 1 :--> ")
    if int(num) is ValueError:
        print("please enter a valid input")
    else:
        num = int(num)
        valid = True
for i in range(1,num+1):
    print("Creating the file number " + str(i))
    file = open("F:\\Learning\\AI ML Learn from Scratch\\DAY 8\\" + str(i) + " Table.txt" , "a")
    for j in range(1,11):
        file.write(str(i) + " * " + str(j) + " = " + str(i*j) + "\n")
    print("done the table of " + str(i) + " had been written in the file ")
    file.close()


    
#A file contains a word “Donkey” multiple times. You need to write a program 
#which replace this word with ##### by updating the same file.  
try:
    file = open("F:\\Learning\\AI ML Learn from Scratch\\DAY 8\\file_donkey.txt" , "r+")
except FileNotFoundError:
    print("the file dosen't exist in the computer")
    exit()
words = file.read()
words = words.lower()
word = "donkey"
changed = 0
words = words.replace(word , "######")
file.seek(0)
file.truncate()
file.write(words)
print("Done")




"""


