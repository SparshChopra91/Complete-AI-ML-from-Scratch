"""
def greater(a,b) :
    if(int(a) > int(b)) :
        return a
    else :
        return b
    
a,b,c = input("please enter the three numbers :-> ").split()
greatest = greater(a,b)
greatest = greater(greatest,c)
print("the greatest of the three numbers is :-> " + str(greatest))

def conversion(celsius) :
    return ((celsius * 9/5) + 32)
celsius = int(input("please enter the temp in celsuis :-> "))
print("the temp " + str(celsius) + " celsius is " + str(conversion(celsius)) + " in Farenhiet")

#recursive function to calculate the sum of n natural numbers 
num = int(input("please enter the number to find the sum of the n natural numbers :-> "))
def sum_of_n(number) :
    if (number == 1) :
        return 1
    total = number + sum_of_n(number-1)
    return total
print("the answer is :-> " + str(sum_of_n(num)))

# recursive function to print a pattern 
num = int(input("please enter the number to print the pattern :-> "))
def pattern(num) :
    if (num == 1):
        print("*")
    print("*" * num , end="")
    pattern(num-1)
pattern(num)

# the program to remove the word from a list 
fruits = ["  apple  ", "banana", "  orange  ", "apple"]
def clean_up(list , word) :
    for i in range(0,len(list)) :
        list[i] = list[i].strip()
    length = len(list)
    while word in list:
        list.remove(word)
    if(length != len(list)) :
        print("the word had been deleted")
    else:
        print("the word is not there in the list")
word = input("enter the word you want to remove :-> ")
clean_up(fruits,word)
print("the new list is " + str(fruits))

"""

