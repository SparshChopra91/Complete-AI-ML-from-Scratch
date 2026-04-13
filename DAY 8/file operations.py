"""
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




"""

valid = False
while (valid is False):
    num = input("please enter the number of table you want to get in files starting from 1 :--> ")
    if(num<0 or type(num) is str):
        print("Please enter a valid input")
    else:
        num = int(num)
        valid = True


for i in range(0,num):
    print("Creating the file number " + str(i+1))
    file = open("F:\\Learning\\AI ML Learn from Scratch\\DAY 8\\" + str(i+1) + " Table.txt" , "w")
    
