import numpy as np 
import matplotlib.pyplot as plt

"""
sales_data = np.array([
    [1,150000,180000,220000,250000], #biryani
    [2,120000,140000,160000,190000], #bites
    [3,200000,230000,260000,300000], #pizza
    [4,180000,210000,240000,270000], #burger
    [5,160000,185000,205000,230000], #chai
])

print("==== the sales data of zomato ====")
print("the shape of the array given is :-  " + str(sales_data.shape))
sales_only = sales_data[: , 1:]
print("printing the matrix with only the sales data :-  " , sales_only)
print("the total sales of all the restaurants are :- " , np.sum(sales_only))
print("the minimum sales per restaurant is :-  " , np.min(sales_only[:,:] , axis=1))
print("the maximum sales of the restaurant in one year :- " , np.max(sales_only[:,:] , axis=0))
print("the average sales of each reatuarants till this year is :-   " , np.mean(sales_data[: , 1:] , axis=1))
print("the cumulative sales of the restaurants are :- " , np.cumsum(sales_only , axis=1))


plt.figure(figsize=(10,10))
plt.plot(np.sum(sales_data[:,1:] , axis=1))
plt.grid(True)
plt.xlabel("Restaurants")
plt.ylabel("Sales")
plt.title("Sales of one year of all restaurants")
plt.show()



vectorized_upper = np.vectorize(str.upper)
resraurants_name = np.array(["chai" , "pizza" , "biryani"])
print("the uppercased array is the of the names of the restaurants are  :- " , vectorized_upper(resraurants_name))



"""


sales_data = np.array([
    [1,150000,180000,220000,250000], #biryani
    [2,120000,140000,160000,190000], #bites
    [3,200000,230000,260000,300000], #pizza
    [4,180000,210000,240000,270000], #burger
    [5,160000,185000,205000,230000], #chai
])

monthly_average = sales_data[:,1:] / 12
print("the monthly average of each of the restaurants are :- " , monthly_average)
plt.figure(figsize=(15,5))
plt.title("showing monthly cummulative average of all the restaurants ")
plt.xlabel=("restaurants")
plt.grid(True)
plt.ylabel=("Cummalative Monthly Average")
plt.plot(np.cumsum(monthly_average[:,:] , axis=1))
plt.show()