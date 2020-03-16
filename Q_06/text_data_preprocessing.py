import nltk
import random
import re
from nltk.tokenize import word_tokenize
from  nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk

def read_data():
    with open("20_newsgroups/alt.atheism/49960","r",errors="ignore") as f:
        text_array = []
        text_array = f.read().splitlines()
        # del text_array[0:28]
        for i in range(len(text_array)):
            text_array[i] = text_array[i].strip()
    return text_array

def lower(text_array):
    
    for i in range(len(text_array)):
        text_array[i] = text_array[i].strip()
        text_array[i] = text_array[i].lower()
    
def remove_special(text_array):

    for i in range(len(text_array)):
        text_array[i] = re.sub("\d+","",text_array[i])
        
        text_array[i] = re.sub("[^a-zA-Z]+"," ",text_array[i])

def tokenizing(text_array):
    for i in range(len(text_array)):
        text_array[i] = word_tokenize(text_array[i])

def stop_words_removal(text_array):
    stop_words = set(stopwords.words('english'))
    
    for i in range(len(text_array)):
        text_array[i] = [w for w in text_array[i] if not w in stop_words]

def lemmetizer_(text_array):
    lemmetizer = WordNetLemmatizer()
    text_array_new = text_array
    for i in range(len(text_array)):
        for j in range(len(text_array[i])):
            text_array_new[i][j] = lemmetizer.lemmatize(text_array_new[i][j])
    return text_array_new
            
def pos_tagging(text_array):
    text_array_new = text_array
    for i in range(len(text_array)):
        text_array_new[i] = nltk.pos_tag(text_array_new[i])

    return text_array_new

def chunking(text_array_pos):
    text_array_chunk = text_array_pos
    for i in range(len(text_array_pos)):
        text_array_chunk[i] = ne_chunk(text_array_pos[i])
    
    return text_array_chunk
if(__name__ == "__main__"):
    
    text_array = read_data()
    i = random.randint(0,len(text_array))
    i=0
    print(text_array[i])
    
    #Converting to lower
    lower(text_array)
    print(text_array[i])

    #Removing numbers and special characters
    remove_special(text_array)
    print(text_array[i])

    #Tokenizing
    tokenizing(text_array)
    print(text_array[i])
        
    #Removing stop words
    stop_words_removal(text_array)
    print(text_array[i])

    #Lemmitizing
    text_array_lemm = lemmetizer_(text_array)
    print(text_array_lemm[i])

    #Parts of speech tagging
    text_array_pos = pos_tagging(text_array)
    print(text_array_pos[i])

    #Chunking
    text_array_chunk = chunking(text_array_pos)
    print(text_array_chunk[i])