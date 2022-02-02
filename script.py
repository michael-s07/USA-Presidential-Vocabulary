import os
import gensim
import spacy
from president_helper import read_file, process_speeches, merge_speeches, get_president_sentences, get_presidents_sentences, most_frequent_words

# get list of all speech files
files = sorted([file for file in os.listdir() if file[-4:] == '.txt'])
#print(files)

# read each speech file
speeches = [read_file(file) for file in files]
#print(speeches)

# preprocess each speech
processed_speeches = process_speeches(speeches)
#print(processed_speeches)

# merge speeches
all_sentences = merge_speeches(processed_speeches)
#print(all_sentences)

# view most frequently used words
most_freq_words = most_frequent_words(all_sentences)
print(most_freq_words)

# create gensim model of all speeches
all_prez_embeddings = gensim.models.Word2Vec(all_sentences, size=96, window=5, min_count=1, workers=2, sg=1)

# view words similar to freedom
similar_to_freedom = all_prez_embeddings.most_similar('freedom', topn = 20)
print(similar_to_freedom)

# get President Roosevelt sentences



# view most frequently used words of Roosevelt



# create gensim model for Roosevelt


# view words similar to freedom for Roosevelt



# get sentences of multiple presidents



# view most frequently used words of presidents



# create gensim model for the presidents


# view words similar to freedom for presidents

