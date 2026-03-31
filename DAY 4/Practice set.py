"""
hindi_dictionary = {
    "नमस्ते": "Hello",
    "धन्यवाद": "Thank you",
    "कृपया": "Please",
    "हाँ": "Yes",
    "नहीं": "No",
    "पानी": "Water",
    "खाना": "Food",
    "दोस्त": "Friend",
    "घर": "House",
    "स्कूल": "School",
    "किताब": "Book",
    "कलम": "Pen",
    "समय": "Time",
    "दिन": "Day",
    "रात": "Night",
    "सूरज": "Sun",
    "चाँद": "Moon",
    "तारा": "Star",
    "आसमान": "Sky",
    "धरती": "Earth",
    "पेड़": "Tree",
    "फूल": "Flower",
    "फल": "Fruit",
    "सब्जी": "Vegetable",
    "बाज़ार": "Market",
    "सड़क": "Road",
    "गाड़ी": "Car",
    "रेल": "Train",
    "हवाई जहाज": "Airplane",
    "पैसा": "Money",
    "काम": "Work",
    "खेल": "Game",
    "जीत": "Win",
    "हार": "Loss",
    "प्यार": "Love",
    "नफरत": "Hate",
    "खुशी": "Happiness",
    "दुख": "Sadness",
    "तेज़": "Fast",
    "धीमा": "Slow",
    "बड़ा": "Big",
    "छोटा": "Small",
    "अच्छा": "Good",
    "बुरा": "Bad",
    "साफ": "Clean",
    "गंदा": "Dirty",
    "गर्म": "Hot",
    "ठंडा": "Cold",
    "नया": "New",
    "पुराना": "Old"
}
hindi_word = input("please enter the hindi word to get the translation :-> ")
english_word = hindi_dictionary.get(hindi_word)
if (english_word is None) :
    print("sorry the word translation is not available ")
else :
    print("the english translation of " + str(hindi_word) + " is " + str(english_word))

numbers = set()
i = 0
while (i<8) :
    numbers.add(int(input("please enter the " + str(i+1) + " number :-> ")))
    i = i+1
print(numbers)
    
normal_set = set()
normal_set.add(18)
normal_set.add("18")
print(normal_set)

random_set = {8,7,12,"Harry" , [1,2]}
#can you change the values inside a list which is contained in a set s 
random_set.remove([1,2])
random_set.add([1,2,3])
print(random_set)
# you cannot have a list as a part of the set

"""

