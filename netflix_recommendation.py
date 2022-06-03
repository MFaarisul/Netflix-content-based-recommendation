import string
import numpy as np
import pandas as pd
import argparse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-title', '--title', required=True,
                    help='Movie or Tv show title')
parser.add_argument('-total_result', '--result', default=5, type=int,
                    help='total result of movie and tv show')
parser.add_argument('-threshold', '--threshold', default=0, type=float,
                    help='Minimum similarity of the movie or tv show')
args = vars(parser.parse_args())


df = pd.read_csv('Data//netflix_titles.csv')

new_df = df[['title', 'type', 'director', 'cast', 'rating', 'listed_in', 'description']]
new_df.set_index('title', inplace=True)

new_df.fillna('', inplace=True)

# For director, cast, and listed_in
# Because there is more than 1 people and categories
# We don't want if people share the same first or last name consider the same person
# or the word that appear in many categories (TV, etc) consider the same category
def separate(texts):
    t = []
    for text in texts.split(','):
        t.append(text.replace(' ', '').lower())
    return ' '.join(t)

def remove_space(texts):
    return texts.replace(' ', '').lower()

def remove_punc(texts):
    return texts.translate(str.maketrans('','',string.punctuation)).lower()


new_df['type'] = new_df['type'].apply(remove_space)
new_df['director'] = new_df['director'].apply(separate)
new_df['cast'] = new_df['cast'].apply(separate)
new_df['rating'] = new_df['rating'].apply(remove_space)
new_df['listed_in'] = new_df['listed_in'].apply(separate)
new_df['description'] = new_df['description'].apply(remove_punc)

new_df['bag_of_words'] = ''
# Combine all the words into 1 column
for i, row in enumerate(new_df.iterrows()):
    string = ''
    for col in new_df.columns:
        if row[1][col] == '':
            continue
        else:
            string += row[1][col] + ' '
            new_df['bag_of_words'][i] = string.strip()

new_df.drop(new_df.columns[:-1], axis=1, inplace=True)


tfid = TfidfVectorizer()
tfid_matrix = tfid.fit_transform(new_df['bag_of_words'])

cosine_sim = cosine_similarity(tfid_matrix, tfid_matrix)

# Later on we will combine with similarity as a column
final_df = df[['title', 'type']]

def recommendation(title, total_result=5, threshold=0):
    # Get the index
    idx = final_df[final_df['title'] == title].index[0]
    # Create a new column for similarity, the value is different for each title you input
    final_df['similarity'] = cosine_sim[idx]
    sort_final_df = final_df.sort_values(by='similarity', ascending=False)[1:total_result+1]
    
    # You can set a threshold if you want to norrow the result down 
    sort_final_df = sort_final_df[sort_final_df['similarity'] > threshold]
    
    # Is the title a movie or tv show?
    movies = sort_final_df['title'][sort_final_df['type'] == 'Movie']
    tv_shows = sort_final_df['title'][sort_final_df['type'] == 'TV Show']
    
    if len(movies) != 0:
        print('Similar Movie(s) list:')
        for i, movie in enumerate(movies):
            print('{}. {}'.format(i+1, movie))
        print()
    else:
        print('Similar Movie(s) list:')
        print('-\n')
        
    if len(tv_shows) != 0:
        print('Similar TV_show(s) list:')
        for i, tv_show in enumerate(tv_shows):
            print('{}. {}'.format(i+1, tv_show))
    else:
        print('Similar TV_show(s) list:')
        print('-')

recommendation(title=args['title'], total_result=args['result'], threshold=args['threshold'])