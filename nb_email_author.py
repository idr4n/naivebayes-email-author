# %%
import pickle
from time import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import GaussianNB

# Loading the Data

try:
    __IPYTHON__
except NameError:
    print("Not in ipython")
    words_file = "./data/word_data.pkl"
    authors_file = "./data/email_authors.pkl"
else:
    print("In ipython")
    words_file = "./ML-projects/NaiveBayes/data/word_data.pkl"
    authors_file = "./ML-projects/NaiveBayes/data/email_authors.pkl"

authors_file_handler = open(authors_file, "rb")
authors = pickle.load(authors_file_handler)
authors_file_handler.close()

words_file_handler = open(words_file, "rb")
words = pickle.load(words_file_handler)
words_file_handler.close()


# Split data into train and test sets

features_train, features_test, labels_train, labels_test = train_test_split(
    words, authors, test_size=0.1, random_state=10)


# Text vectorization--go from strings to lists of numbers

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test)


# Reduce the amount of features because the text has too many features (words)
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(features_train, labels_train)
features_train = selector.transform(features_train).toarray()
features_test = selector.transform(features_test).toarray()


# Info on the data
print("\nNo. of Chris training emails:", sum(labels_train))
print("No. of Sara training emails:", len(labels_train)-sum(labels_train))


# Train the NB model

t0 = time()
model = GaussianNB()
model.fit(features_train, labels_train)
print(f"\nTraining time: {round(time()-t0, 3)}s")
t0 = time()
score_train = model.score(features_train, labels_train)
print(f"Prediction time (train): {round(time()-t0, 3)}s")
t0 = time()
score_test = model.score(features_test, labels_test)
print(f"Prediction time (test): {round(time()-t0, 3)}s")

print("\nTrain set score:", score_train)
print("Test set score:", score_test)
