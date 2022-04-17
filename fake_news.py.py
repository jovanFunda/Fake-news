from sklearn.model_selection import train_test_split
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

lancaster = LancasterStemmer()

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

# Vraca tekstove sa ociscenim recima, manjim brojem reci
def clean_text(series):
    cleaned_text = []
    stop_punc = set(stopwords.words('english')).union(set(punctuation))
    for doc in series:
      words = wordpunct_tokenize(doc)
      words_lower = [w.lower() for w in words]
      words_filtered = [w for w in words_lower if w not in stop_punc]
      words_stemmed = [lancaster.stem(w) for w in words_filtered]
      cleaned_text.append(words_stemmed)
    return cleaned_text

# Koliko najcesce koriscenih reci zelimo u nasem vokabularu
def get_vocab(word_count):
    voc = []
    for text in cleaned_texts:
      for w in text:
          if len(w) > 2:
              voc.append(w)
    
    final_vocab = []
    words = Counter(voc).most_common(word_count)
    for tup in words:
        final_vocab.append(tup[0])
    return final_vocab

def numocc_score(word, doc):
  return doc.count(word)


# Vraca koliko se koja puta rec ponavlja u jednom tekstu
def create_feature(filtered_text, vocab):
  var = np.zeros((len(filtered_text), len(vocab)), dtype=np.float32)
  for doc_idx in range(len(filtered_text)):
      doc = filtered_text[doc_idx]
      for word_idx in range(len(vocab)):
        word = vocab[word_idx]
        cnt = numocc_score(word, doc)
        var[doc_idx][word_idx] = cnt

  return np.asarray(var)

def get_predicted_values(testX, model):
  pred = []
  for i in range(len(testX)):
      test = testX[i]
      prediction = model.predict(test)
      pred.append(prediction)
  return pred


def valid_invalid_words(features, labels, vocab):
  valid_words = np.zeros((len(features[0])), dtype=np.float32)
  invalid_words = np.zeros((len(features[0])), dtype=np.float32)
  for i in range(len(features)):
    if labels[i] == 1:
      for j in range(len(features[0])):
        valid_words[j] += features[i][j]
    else:
      for j in range(len(features[0])):
        invalid_words[j] += features[i][j]
  
  valid_vocab = zip(vocab, valid_words)
  invalid_vocab = zip(vocab, invalid_words)

  return [list(valid_vocab), list(invalid_vocab)]

# Ucitavamo fajl i izbacuje sve redove koje sadrze NaN vrednosti
df = pd.read_csv(r'data\fake_news.csv')
df.dropna(subset = ["title", "author", "text"], inplace=True)

# Cistimo tekst od viska reci, smanjujemo njihovu velicinu
cleaned_texts = clean_text(df["text"])

# Vraca 10.000 najcesce koriscenih reci u tekstovima
vocab = get_vocab(10000)

# Koliko se puta reci ponavljaju u jednom tekstu
# Velicine broj tekstova * broj reci u vokabularu
X = create_feature(cleaned_texts, vocab)

y = np.asarray(df['label'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNaiveBayes(nb_classes=2, nb_words=10000, pseudocount=1)
model.fit(X_train, y_train)

# Testiramo nas model
predicted = get_predicted_values(X_test, model)

conf_matrix = confusion_matrix(y_test, predicted)

# Vraca 2 dictionary objekta velicine broja reci u vokabularu
# Predstavlja koliko se puta reci pojavljuju u pouzdanim / nepouzdanim tekstovima
valid_invalid_list = valid_invalid_words(X, y, vocab)

top_5_valid_words = (sorted(list(valid_invalid_list[0]), key=lambda x: x[1], reverse=True))[:5]
top_5_invalid_words = (sorted(list(valid_invalid_list[1]), key=lambda x: x[1], reverse=True))[:5]


valid_words = valid_invalid_list[0]
invalid_words = valid_invalid_list[1]
LR = defaultdict(float)

# Racunanje LR metrike iz zadatka za svaku rec
for i in range(len(X[0])):
  if valid_words[i][1] < 10 or invalid_words[i][1] < 10 or invalid_words[i][1] == 0:
    LR[valid_words[i][0]] = -1.0
  else:
    LR[valid_words[i][0]] = valid_words[i][1] / invalid_words[i][1]


LR = {key:val for key, val in LR.items() if val != -1.0}
LR = dict(sorted(LR.items(), key=lambda item: item[1]))

lr_list = list(LR)
least_5_LR = lr_list[:5]
top_5_LR = lr_list[-5:]

# (tp + tn) / (fp + fn + tp + tn)
print("Accuracy:", accuracy_score(y_test, predicted))
# tp / (tp + fp)
print("Precision score:", precision_score(y_test, predicted))
print("Matrica konfuzije: \n", conf_matrix)
print("Top 5 valid:", top_5_valid_words)
print("Top 5 invalid:", top_5_invalid_words)
print("LR least:", least_5_LR, "LR top: ", top_5_LR)


# Top 5 valid: [('trump', 17347.0), ('clinton', 17217.0), ('stat', 16664.0), ('peopl', 14556.0), ('new', 13647.0)]
# Top 5 invalid: [('said', 68014.0), ('trump', 36108.0), ('stat', 30337.0), ('new', 29384.0), ('would', 22665.0)]
# Ljudi uglavnom vole da raspravljaju o politici u ovom datasetu i imaju opredeljeno misljenje o politicarima

# Metrika LR oznacava koliko je verovatnije da se rec pojavljuje u pouzdanim tekstovima, nego u nepouzdanim

# LR least: ['kushn', 'tourna', 'rio', 'milo', 'devo'] 
# U nepouzdanim tekstovima se uglavnom nalaze najvise reci stranog porekla

# LR top:  ['extraterrest', 'das', 'que', 'infow', 'www']
# Dok se u pouzdanim tekstovima uglavnom nalaze reci sa konkretnim internet pojmovima