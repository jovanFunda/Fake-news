from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import numpy as np
import pandas as pd
from collections import Counter

class MultinomialNaiveBayes:
  def __init__(self, nb_classes, nb_words, pseudocount):
    self.nb_classes = nb_classes
    self.nb_words = nb_words
    self.pseudocount = pseudocount
  
  def fit(self, X, Y):
    nb_examples = X.shape[0]

    # Koja je verovatnoca klasa unutar celog skupa
    self.priors = np.bincount(Y) / nb_examples

    # Racunamo broj pojavljivanja svake reci u svakoj klasi
    occs = np.zeros((self.nb_classes, self.nb_words))
    for i in range(nb_examples):
      c = Y[i]
      for w in range(self.nb_words):
        cnt = X[i][w]
        occs[c][w] += cnt
    
    # Racunamo P(Rec_i|Klasa) - likelihoods
    # Vraca vrednosti koje pokazuju za svaku rec koliko je verovatno da se sadrzi ukoliko je odredjena klasa u pitanju
    self.like = np.zeros((self.nb_classes, self.nb_words))
    for c in range(self.nb_classes):
      for w in range(self.nb_words):
        up = occs[c][w] + self.pseudocount
        down = np.sum(occs[c]) + self.nb_words*self.pseudocount
        self.like[c][w] = up / down
          
  def predict(self, bow):
    # Racunamo P(Klasa|bow) za svaku klasu
    probs = np.zeros(self.nb_classes)
    for c in range(self.nb_classes):
      prob = np.log(self.priors[c])
      for w in range(self.nb_words):
        cnt = bow[w]
        prob += cnt * np.log(self.like[c][w])
      probs[c] = prob
    # Trazimo klasu sa najvecom verovatnocom
    prediction = np.argmax(probs)
    return prediction


df = pd.read_csv(r'fake_news.csv')
df.dropna(subset = ["title", "author", "text"], inplace=True)

cleaned_text = []
lancaster = LancasterStemmer()

stop_punc = set(stopwords.words('english')).union(set(punctuation))
for doc in df["text"]:
    words = wordpunct_tokenize(doc)
    words_lower = [w.lower() for w in words]
    words_filtered = [w for w in words_lower if w not in stop_punc]
    words_stemmed = [lancaster.stem(w) for w in words_filtered]
    cleaned_text.append(words_stemmed)

vocab = []

for text in cleaned_text:
    for w in text:
        if len(w) > 2:
            vocab.append(w)

final_vocab = []
words = Counter(vocab).most_common(10000)
for tup in words:
    final_vocab.append(tup[0])

def numocc_score(word, doc):
  return doc.count(word)

X = np.zeros((len(cleaned_text), len(final_vocab)), dtype=np.float32)
for doc_idx in range(len(cleaned_text)):
     
    doc = cleaned_text[doc_idx]
    for word_idx in range(len(final_vocab)):
      word = final_vocab[word_idx]
      cnt = numocc_score(word, doc)
      X[doc_idx][word_idx] = cnt
    
print()

y = df['label'].values

X = np.asarray(X)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNaiveBayes(nb_classes=2, nb_words=10000, pseudocount=1)
model.fit(X_train, y_train)

predicted = []
for i in range(len(X_test)):
    test = X_test[i]
    prediction = model.predict(test)
    predicted.append(1 if y_test[i] == prediction else 0)
    print('Predicted class ', i , prediction)

print("accuracy:", sum(predicted) / len(predicted))