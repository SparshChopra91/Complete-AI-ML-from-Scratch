"""
#program 1
try :
    file1 = open("F:\\Learning\\AI ML Learn from Scratch\\Advanced Python 1\\files 1.py")
    print("the file 1 had been opened")
except FileNotFoundError:
    print("the file is not there at the given location")
try :
    file2 = open("F:\\Learning\\AI ML Learn from Scratch\\Advanced Python 1\\files 2.py")
    print("the file 2 had been opened")
except FileNotFoundError:
    print("the file is not there at the given location")
try :
    file3 = open("F:\\Learning\\AI ML Learn from Scratch\\Advanced Python 1\\files 3.py")
    print("the file 3 had been opened")
except FileNotFoundError:
    print("the file is not there at the given location")
print("the program had been ended")



#program 2
sample_list = ["Python", 7, 3.14, "Donkey", True, 100, "Vector", 0, False, "Microsoft"]
for i , item in enumerate(sample_list , start=0):
    if(i in [3,5,7]):
        print(i,item)



#program 3
num1 = int(input("please enter the first number :---->  "))
num2 = int(input("please enter the second number :---> "))
try :
    print("the divison result is " + str(num1/num2))
except ZeroDivisionError:
    print("the second number cannot be zero")




"""
