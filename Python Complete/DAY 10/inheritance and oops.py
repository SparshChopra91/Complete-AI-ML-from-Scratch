"""


class animal():
    species = "larbrador"
    age = 0
    def age_set(self , age):
        self.age = age
        print("the age of the animal had been set :---> " + str(self.age))
class pet(animal):
    name = "NO Name"
    owner = "No Input"
    def set_name(self,name):
        self.name = name 
        print("the name of the pet had been set to ----> " + str(self.name))
class dog(pet):
    @staticmethod
    def bark():
        print("BOW!! BOW!! the dog barked")

        


class employee():
    name = "default"
    salary = 0
    increment = 5000
    def __init__(self , name , salary):
        self.name = name 
        self.salary = salary
    @property
    def salary_after_increment(self):
        return self.salary + self.increment
    @salary_after_increment.setter
    def salary_after_increment(self , salary):
        if(salary < 0 or salary < self.salary):
            print("this is invalid")
            return
        self.increment = salary - self.salary
        

raman = employee("raman" , 10000)
print(raman.salary_after_increment)
raman.salary_after_increment = 12000


#🚀 💀 ADVANCED INHERITANCE + PROPERTY QUESTION
class employee():
    name = ""
    base_salary = 0
    def __init__(self , name , salary):
        self.name = name 
        self.base_salary = salary
    @property
    def total_salary(self):
        return self.base_salary
    @total_salary.setter
    def total_salary(self , salary):
        if(salary < 0 or salary < self.total_salary):
            print("please enter teh valid input only ")
            return
        self.base_salary = salary
        print("the new total salary had been updated")
class manager(employee):
    bonus = 0
    def __init__(self, name, salary , bonus):
        super().__init__(name, salary)
        self.bonus = bonus
    @property
    def total_salary(self):
        return self.base_salary + self.bonus
    @total_salary.setter
    def total_salary(self , salary):
        if(salary < 0 or salary < self.total_salary):
            print("please enter teh valid input only ")
            return
        self.bonus = salary - self.base_salary
        print("the new salary had been updated ")
class developer(employee):
    overtime_hours = 0
    rate_per_hour = 250
    def __init__(self, name, salary , overtime_hours):
        super().__init__(name, salary)
        self.overtime_hours = overtime_hours
    @property
    def total_salary(self):
        return int(self.base_salary + (self.rate_per_hour * self.overtime_hours))
    @total_salary.setter
    def total_salary(self , salary):
        if(salary < 0 or salary < self.total_salary):
            print("please enter teh valid input only ")
            return
        self.overtime_hours = int((salary - self.base_salary)/self.rate_per_hour)
        print("the new salary had been updated")
class techlead(manager):
    team_size = 0
    def __init__(self, name, salary, bonus , team_size):
        super().__init__(name, salary, bonus)
        self.team_size = team_size
    @property
    def total_salary(self):
        return (self.base_salary + self.bonus + (self.team_size * 1000))
    @total_salary.setter
    def total_salary(self ,salary):
        if(salary < 0 or salary < self.total_salary):
            print("please enter teh valid input only ")
            return
        self.bonus = (salary - self.base_salary - (self.team_size * 1000 ))
        print("the new salary had been updated")



#smart vector system 
class vector():
    @staticmethod
    def defination_of_class():
        print("this is the vector system class in which it does all operations on vectors")
    def __init__(self , list):
        if (len(list) <= 0):
            print("please enter a valid list")
            return
        self.list = list
    def __len__(self):
        return len(self.list)
    def __str__(self):
        if(len(self.list) <= 0):
            print("the object dosen't have the values for the vector")
            return
        elif (len(self.list) == 2):
            return str((str(self.list[0]) + "i + " + str((self.list[1])) + "j"))
        elif(len(self.list) == 1):
            return str((str(self.list[0])) + "i")
        elif(len(self.list) == 3):
            return str((str(self.list[0]) + "i + " + str((self.list[1])) + "j + " + str((self.list[2])) + "k"))
        else :
            return (str(self.list))
    def __add__(self, other):
        if(len(self.list) != len(other.list)):
            print("only same dimension vector can be added")
            return
        list_temp = []
        for i in range(0,len(self.list)):
            list_temp.append((self.list[i] + other.list[i]))
        return vector(list_temp)
vector1 = vector([1,2,3])
vector2 = vector([1,2])
vector3 = vector([4,5,6])
print("the length of the vector1 is " + str(vector1.__len__()))
print(vector1)
print(vector1 + vector1)




"""
