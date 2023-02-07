import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
stemmer = SnowballStemmer('english')
from gensim.parsing.preprocessing import preprocess_string, strip_punctuation, strip_numeric



'''
Must preprocess data.
Remove the stopwords (the, and, ...)
Lemmatize each word (rocks -> rock)
Remove words less than 3 letters
'''
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v')))
    return result



'''
Copy data to new dataframe to work with
'''
df = pd.read_csv('./tweets.csv')
data = df[['Tweet']]
data['index'] = data.index
documents = data


'''
Look at how one sentence is prepreocessed
'''
doc_sample = documents[documents['index'] == 10000].values[0][0]
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print(preprocess(doc_sample))


'''
Apply proeprocess function to all tweets in df
'''
processed_docs = documents['Tweet'].map(preprocess)

'''
gives value to each word based on how many tweets it appears in tweets
'''
dict1 = gensim.corpora.Dictionary(processed_docs)
count = 0
for k,v in dict1.iteritems():
    print(k,v, dict1.dfs[k])
    count += 1
    if count > 10:
        break
        
'''
removes extremes from dictionary
removes if appears in less than 5 tweets
removes if appears in more than 70% of tweets
only keep keep_n most frequent tokens
'''
dict1.filter_extremes(no_below=5, no_above=0.7, keep_n=100000)




'''
Convert list of words into a list of (token_id, token_count)
'''
bow_corpus = [dict1.doc2bow(doc) for doc in processed_docs]



'''
Estimates Latent Dirichlet Allocation model parameters
Similar to k-means cluster but meant for bag of words
Does two passes through data
Categorizes into 20 topics
'''
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=dict1, passes=2, workers=4)




'''
print out topics and the most common words in each
'''
for idx, topic in lda_model.show_topics(formatted=False, num_words= 20):
    print('Topic: {} \nWords: {}'.format(idx, [w[0] for w in topic]))
    
