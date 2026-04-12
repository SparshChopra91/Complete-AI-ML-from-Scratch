"""
# print the multiplication table of the given number 
num = int(input("please enter the number you want to get the multiplication table for :-> "))
i = 0
while (i<11) :
    print(str(num) + " * " + str(i) + " = " + str(num*i))
    i += 1
print("done")

# write the program to greet only the person name starting from s 
temp_list = ["Harry", "Soham", "Sachin", "Rahul"]
i = 0
while (i < len(temp_list)) :
    temp = temp_list[i].lower()
    i +=1
    if(temp.startswith("s")) :
        print("hello " + temp)
    else :
        continue
print("done")

#program to find the sum of first n natural numbers sum 
num = int(input("please enter the number :-> "))
i = num-1
total = num
while (i>0) :
    total = total + i
    i = i-1
print("the sum is the :-> " + str(total))

# star pattern one 
num = int(input("please enter the rows number to print :-> "))
for i in range(0,num) :
    num1 = 0
    while (num1 < i+1) :
        print("*" , end="")
        num1 += 1
    print("\n")
print("\n done")

# star pattern second
num = int(input("please enter the rows number to print :-> "))
for i in range(0,num) :
    num1 = 0
    while (num1 < (num-i-1)) :
        print(" ",end="")
        num1 += 1
    num1 = 0
    while (num1 < ((i+1)*2)-1) :
        print("*" , end="")
        num1 += 1
    print("\n")
print("\n done")


# star pattern second
num = int(input("please enter the rows number to print :-> "))
for i in range(0,num) :
    print(" " * (num-i-1),end="")
    print("*" * (((i+1)*2)-1), end="")
    print("\n")
print("\n done")

"""
