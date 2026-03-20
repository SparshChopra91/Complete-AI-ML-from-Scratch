print("hello world i am starting to learn the python")
import pyjokes
#just trying to use the import function 
""" hello there 
this is the multi line comment use case """
joke = pyjokes.get_joke()
print(joke + "\n")
import pyttsx3
pyttsx3.speak(joke)
pyttsx3.speak("here comes a new joke\n")
joke = pyjokes.get_joke()
print(joke)
pyttsx3.speak(joke)