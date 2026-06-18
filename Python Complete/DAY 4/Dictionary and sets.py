marks = {
    "rohit" : 40,
    "mohit" : 100,
    "rohan" : 95,
}
print(type(marks))
print("the marks of rohan are :-> " + str(marks["rohan"]))
marks.update({"rohit" : 75})
print(str(marks.items()))