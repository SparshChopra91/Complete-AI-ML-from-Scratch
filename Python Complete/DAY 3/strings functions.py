a = "Hello My name is Sparsh Chopra "
print("the length of the string is :->  " + str(len(a)))
print("whether the string ends with Chopra or not :->  " +  str(a.endswith("Chopra")))
print("the number of occurance of a in the string are :->  " + str(a.count("a")))
print("finding the index of the word Sparsh :->  " + str(a.find("Sparsh")))  # only the first occurance no other check if found one 
print("replacing the word using the strings build in functions :--> " + str(a.replace("Hello" , "Hellooooooo")))
print(a)

#note that the replace function will not change the original string , the strings are immutable in python 

b = " Testing The escape sequence \"Characters\" "
print(b)