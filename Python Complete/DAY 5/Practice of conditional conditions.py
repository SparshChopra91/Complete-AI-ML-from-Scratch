"""
# write the program to find the greatest of the 4 numbers entered by the user 
print("this program will print the greatest of the 4 numbers entered by you :-- ")
i = 0
while(i<4) :
    if(i==0) :
        max_num = int(input("please enter the first number :-> "))
    else :
        temp = int(input("please enter the next number :-> "))
        if(temp > max_num) :
            max_num = temp
    i = i+1
print("the greatest number among all is :-> " + str(max_num))



# Write a program to find out whether a student has passed or failed if it requires a
#total of 40% and at least 33% in each subject to pass. Assume 3 subjects and
#take marks as an input from the user.

marks = []
i = 0
while (i<3) :
    marks.append(int(input("please enter the " + str(i+1) + " subject marks :-> ")))
    if(marks[i] < 33) :
        print("you haven't passed this exam so fail")
        exit()
    i = i+1
total_perc = sum(marks) / len(marks)
if(total_perc < 40) :
    print("sorry you have failed as 40 criteria not accomplished")
else :
    print("you have passed the exam with the " + str(total_perc) + " percentage of marks")

    


#Write a program to find whether a given username contains less than 10
#characters or not.
username = input("please enter the username for checking :-> ")
if (len(username) < 10 ) :
    print("the username contained less than the 10 characters")
else : 
    print("the username dosen't contain less than 10 characters ")

    



#Write a program which finds out whether a given name is present in a list or not.
names = ["Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah", "Ian", "Julia", "Kevin", "Luna", "Milo", "Nora", "Oscar"]
user = input("please enter the name to check in the list :-> ")
if (user in names) :
    print("yes , the name entered by you is in the list")
else :
    print("no , the name entered by you is not in the list")

    


    
#Write a program to find out whether a given post is talking about “Harry” or not.
post = input("please enter the content of the post :-> ")
if("harry" in post.lower()) :
    print("yes , the post is talking about the harry")
else :
    print("No , the post is not talking about harry")

"""
