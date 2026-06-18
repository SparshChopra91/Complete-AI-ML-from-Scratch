"""

#create  the class programmer for storing info of programmers at microsoft
class employee:
    company = "Microsoft"
    def __init__(self , salary , name , age , dept):
        self.salary = salary
        self.name = name.lower()
        self.age = age
        self.dept = dept
    def print_info(self):
        print("the name of the employee is " + self.name)
        print("the salary of the employee is " + str(self.salary)) 
harsh = employee(50000,"Harsh",23,"HR")
harsh.print_info()


#calculator class and able to calculate the square , cube and sq root 
class calculator:
    number = 0
    def __init__(self , number):
        self.number = number
    def square(self):
        return (self.number*self.number)
    def cube(self):
        return(self.number*self.number*self.number)
    def sq_root(self):
        return(self.number**0.5)
number1 = calculator(36)
print(number1.square())
print(number1.cube())
print(number1.sq_root())


#calculator class and able to calculate the square , cube and sq root 
class calculator:
    number = 0
    def __init__(self , number):
        self.number = number
    @staticmethod
    def hello():
        print("hello there")
    def square(self):
        return (self.number*self.number)
    def cube(self):
        return(self.number*self.number*self.number)
    def sq_root(self):
        return(self.number**0.5)
number1 = calculator(36)
print(number1.square())
print(number1.cube())
print(number1.sq_root())




"""

#🚀 💀 NEXT LEVEL VERSION (UPGRADED TRAIN SYSTEM)
try :
    file_booking = open("F:\\Learning\\AI ML Learn from Scratch\\DAY 9\\booking.txt" , "r+")
except FileNotFoundError:
    print("the file dosen't exists in the computer creating one :----->  ")
    file_booking = open("F:\\Learning\\AI ML Learn from Scratch\\DAY 9\\booking.txt" , "w")
class train:
    train_name = "Rajni Express"
    total_seats = 5
    available_seats = 5
    fare = 500
    booked_seats = {}

    def book_ticket(self , name):
        self.name = name.lower()
        if self.name in train.booked_seats:
            print("you had already booked the train ticket at " + str(train.booked_seats[self.name]))
            return
        if(train.available_seats == 0):
            print("sorry all the tickets had been booked , no seats now ")
            return
        if(train.available_seats > 0):
            self.seat_number = train.available_seats
            train.available_seats -= 1
        booking = str(self.name) + ":  " + str(self.seat_number)
        file_booking.write(booking + "\n")
        train.booked_seats[self.name] = self.seat_number
        print("you have successfully made the booking in the train " + booking)
    
    @staticmethod
    def get_status():
        print("the total seats in the train are " + str(train.total_seats))
        print("the total seats available in the train are " + str(train.available_seats))
        print("the total seats booked in the train are " + str(train.booked_seats))
    @staticmethod
    def get_fare():
        print("the fare of the ride in the train is :-> " + str(train.fare))

    def cancel_ticket(self , name):
        self.name = name.lower()
        if self.name not in train.booked_seats:
            print("you haven't booked any seats in the train ")
            return
        words = "."
        final = ""
        file_booking.seek(0)
        while True:
            if (words == ""):
                break
            words = file_booking.readline()
            if self.name in words:
                continue
            else:
                if(final == ""):
                    final = words
                else:
                    final = final + "\n" + words
        file_booking.seek(0)
        file_booking.truncate()
        file_booking.write(final)
        print("your ticket had been successfully canceled " + str(self.name) + " and the seat number " + str(train.booked_seats[self.name]))
        del train.booked_seats[self.name]
        train.available_seats += 1