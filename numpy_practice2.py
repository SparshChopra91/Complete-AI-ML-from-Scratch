import numpy as np 

"""

where_array = np.where(test_array2 > 5)
mask = test_array2 > 5
normal_array = np.array(test_array2[test_array2 > 5])
mask_array = np.array(test_array2[mask])

print(test_array2[where_array])
print(normal_array)
print(mask_array)


where1 = np.where((test_array > 5) & (test_array < 9))
where2 = np.where(test_array>5 , "true" , "false")
print(test_array[where1])
print(where2)



test_array = np.array([1,2,3,4,5,6])
test_array_2 = np.array([11,12,13,14,15,16])
combined_array = np.concatenate((test_array,test_array_2))
print("the combined arrays are:- " , str(combined_array))
vstack_array = np.vstack((test_array,test_array_2))
if(test_array.shape == test_array_2.shape):
    print("the arrays can be combined by the vstack :- " + str(vstack_array)) 
else:
    print("the arrays shape dosen't match so can be added ")

    


"""

