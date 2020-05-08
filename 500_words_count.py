# from nltk.corpus import names
# print(names.words()[:10])
# print(len(names.words()))



# A box of Text Model

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer

def letters_only(astr):
    return astr.isalpha()

cv = CountVectorizer(stop_words="english", max_features=500)
groups = fetch_20newsgroups()
cleaned = []
all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

for post in groups.data:
    cleaned.append(' '.join([lemmatizer.lemmatize(word.lower())
    for word in post.split()
    if letters_only(word)
    and word not in all_names]))

transformed = cv.fit_transform(cleaned)
from sklearn.decomposition import NMF
nmf = NMF(n_components = 100,
          random_state = 43).fit(transformed)
for topic_idx, topic in enumerate(nmf.components_):
    label = '{}: '.format(topic_idx)
    print(label, " ".join([cv.get_feature_names()[i]
                       for i in topic.argsort()[:-9:-1]]))

# print(cv.get_feature_names())

# sns.distplot(np.log(transformed.toarray().sum(axis=0)))

# plt.xlabel('Log Count')
# plt.ylabel('Frequency')
# plt.title('Distribution Plot of 500 Word Counts')
# plt.show()

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 20)
km.fit(transformed)

labels = groups.target

plt.scatter(labels, km.labels_)
plt.xlabel('News Group')
plt.ylabel('Cluster')
plt.title('Distribution Plot of 500 Word Counts')
plt.show()
