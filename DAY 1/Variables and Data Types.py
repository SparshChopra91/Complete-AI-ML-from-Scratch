a=1
b=2
c=a+b
d=True
import pyttsx3
#pyttsx3.speak("the answer is" + str(c))
print(c)
e = type(d)
print(e)
f = "12.5"
print(int(float(f)))

aa , bb = input("please enter 2 numbers to add :---> ").split()
print("the final answer is :-> " + str(int(aa) + int(bb)))
if (int(aa) > int(bb)) :
    print("the first number " + aa + " is bigger than the second number")
else :
    print("the second number " + bb + " is bigger than the first number")
cc = float(aa) % float(bb)
print("the remainder we get when we divide the first number with the second number is " + str(cc))