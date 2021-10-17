import pickle
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import json
import random
import tensorflow as tf
import tflearn
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

# words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = [lemmatizer.lemmatize(w.lower()) for w in words if w != "?"]

# words.extend([lemmatizer.lemmatize(w.lower()) for w in words if w != "?"])
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    # wrds = [stemmer.stem(w.lower()) for w in doc]
    wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]

    # wrds.extend([lemmatizer.lemmatize(w.lower()) for w in doc])

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_verbose=3)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True,)
print(model.get_weights(net.W))
model.save("model.tflearn")
