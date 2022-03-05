#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import required packages

import pandas as pd
import numpy as np
import nltk
import os.path
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
from nltk.tokenize import word_tokenize

# In[2]:


# convert movies.csv file to dataframe

movies = pd.read_csv('MovieLens/movies.csv')
print(movies.shape)
print(movies.head())


# In[3]:


# find out how many unique genres there are

series_genres = movies['genres'].str.split('|')
all_genres = []
for i in range(movies.shape[0]):
    all_genres += series_genres[i]

list_unique_genres = list(set(all_genres))
print(list_unique_genres)
print('number of genres =', len(list_unique_genres))


# In[4]:


# remove the rows without genre, "no genre listed"

movies = movies.drop(movies.loc[movies['genres'] == '(no genres listed)'].index)
print(movies.shape)


# In[5]:


# separate multiple-genre movies in multiple rows with a single genre

movies['genres'] = movies['genres'].str.split('|')
movies_single_genre = movies.explode('genres', ignore_index=True)
print(movies_single_genre.iloc[:10, :])


# In[6]:


# put all movie titles of a certain genre in a row using groupby

grouped_movies = movies_single_genre.groupby("genres")
grouped_movies_genre = grouped_movies["title"].apply(list)
grouped_movies_genre = grouped_movies_genre.reset_index()
print(grouped_movies_genre)


# In[7]:


# create a dictionary with genres as keys and titles of each genres as value in a way that each key has a string of all
# its corresponding titles

def str_per_genre(df):
    
    dic_titles_per_genre = dict()
    stop_words = set(stopwords.words('english'))  # to remove prepositions, auxiliary verbs, and the like
    words = set(nltk.corpus.words.words())  # to ignore nonsense words
    
    for i in range(df.shape[0]):
        list_titles = df.iloc[i,1]
        str_titles = ' '.join(list_titles)
        list_words_per_genre = [w for w in nltk.word_tokenize(str_titles) if w.lower() in words if w.isalpha()
                                if not w.lower() in stop_words]
        dic_titles_per_genre[df.iloc[i, 0]] = ' '.join(list_words_per_genre)
        
    return dic_titles_per_genre


dict_titles_per_genre = str_per_genre(grouped_movies_genre)


# In[11]:


# save all titles of each genres as text files

# my_path = C:\Users\mahmo\PycharmProjects\pythonProject\MovieLens
save_path = input("Please enter your directory:\n")

for i, key in enumerate(dict_titles_per_genre):
    name_of_file = 'titles_' + str(i)  # dict_titles_per_genre
    completeName = os.path.join(save_path, name_of_file + ".txt")         
    file = open(completeName, "w")
    toFile = dict_titles_per_genre[key]
    file.write(toFile)
    file.close()

