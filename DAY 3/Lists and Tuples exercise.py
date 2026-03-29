"""
fruits = []
i = 0
while(i < 7) :
    fruits.append(input("please enter the fruit :->  "))
    i = i+1
print(fruits)


marks = []
i = 0
while(i < 6) :
    marks.append(input("please enter the marks of " + str(i+1) + " student :->  "))
    i = i+1
marks.sort()
print(marks)


list1 = [11 , 45 , 9 , 58]
total = sum(list1)
print("the sum of the lists is :-> " + str(total))

"""

a = (7,0,8,0,0,9)
count = a.count(0)
print("calculating all the zeros in the tuple :-> " + str(count))