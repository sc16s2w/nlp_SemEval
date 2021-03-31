import datetime

import pandas as pd
import csv
import string
import numpy as np
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import tensorflow as tf

from datetime import datetime
from sklearn.metrics import f1_score as sklearn_f1_score
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

LAMBDA_L2_REG = 0.00001
LOG_DIR = "./model_output"
LEARNING_RATE = 0.001
BATCH_SIZE = 128
TRAINABLE_EMBEDDINGS = False
EPOCHS = 100
LSTM_NEURONS = 256
NEURONS_HIDDEN_LAYER_1 = 128
RNN_LAYERS = 3
DROPOUT_KEEP_PROBABILITY = 0.5
NEURONS_SOFTMAX = 2

def read_from_tsv(file_path: str, column_names: list) -> list:
    csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(file_path, "r") as wf:
        reader = csv.DictReader(wf, fieldnames=column_names, dialect='tsv_dialect')
        datas = []
        for row in reader:
            data = dict(row)
            datas.append(data)
    csv.unregister_dialect('tsv_dialect')
    return datas

def shuffle_batch(X, y, batch_size):
    """
    Create batches.
    :param X: Features
    :param y: Labels
    :param batch_size:
    :return:
    """
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        print(batch_idx)
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, np.reshape(y_batch, (y_batch.shape[0],))


#
class DataReader:
    def __init__(self, file_path, column_names, sub_task=None):
        self.file_path = file_path
        self.column_names = column_names
        self.sub_task = sub_task

    def get_data(self):
        csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
        with open(self.file_path, "r") as wf:
            reader = csv.DictReader(wf, fieldnames=self.column_names, dialect='tsv_dialect')
            datas = []
            labels = []
            i = 0
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                data = dict(row)
                datas.append(data["tweet"])
                if self.sub_task == "A":
                    label = self.label_to_int(data["subtask_a"])
                    labels.append(label)
                if self.sub_task == "B":
                    label = self.label_to_int(data["subtask_b"])
                    labels.append(label)
                if self.sub_task == "C":
                    label = self.label_to_int(data["subtask_c"])
                    labels.append(label)
        csv.unregister_dialect('tsv_dialect')
        return datas, labels

    def label_to_int(self, label):
        if label == "OFF":
            label = 1
        elif label == "NOT":
            label = 0
        elif label == "TIN":
            label = 1
        elif label == "UNT":
            label = 0
        elif label == "GRP":
            label = 0
        elif label == "IND":
            label = 1
        elif label == "OTH":
            label = 2
        return label


class Preprocess:
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data
        print("preprocess the data")
        self.remove_emoji(data)
        self.remove_unnecessary(data)

    # remove the emoji
    def remove_emoji(self, data):
        for i, element in enumerate(data):
            self.data[i] = emoji.demojize(element)
        return data

    # remove stop words, redundant information, tokeizer
    def remove_unnecessary(self, data):
        tokenizer = RegexpTokenizer(r'\w+')
        lemmatizer = WordNetLemmatizer()
        for i, element in enumerate(data):
            element = element.replace('@USER', '')
            element = element.replace('URL', '')
            element = tokenizer.tokenize(element)
            stop_words = set(stopwords.words('english'))
            element = [w for w in element if w not in stop_words]
            element = [w for w in element if not w.isdigit()]
            element = [w for w in element if w not in string.punctuation]
            self.data[i] = [lemmatizer.lemmatize(w, pos='v') for w in element]
        return data

    # change it into the right format
    def change_data(self, data, labels):
        d1 = {}
        for i, element in enumerate(data):
            d1[i] = element
        label = np.array(labels).reshape(len(labels), 1)
        series_data = pd.Series(data=d1, index=d1.keys())
        return series_data, label

    def prepare_data(self, X_train, X_test, max_tweet_length):
        """
        Prepare data by tokenizing it.
        :param X_train: Train data as an ndarray.
        :param X_test: TestA data as an ndarray.
        :param max_tweet_length: A maximum length of a tweet.
        :return: Padded train/test data, mapping of words to the number of texts they appeared, mapping of words to indices
        """

        tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
        tokenizer.fit_on_texts(X_train.ravel())
        train_words_to_indices = tokenizer.texts_to_sequences(X_train.ravel())
        test_words_to_indices = tokenizer.texts_to_sequences(X_test.ravel())

        # Add zeroes to to the tweet, if its length less than max_tweet_length.
        train_padded = tf.keras.preprocessing.sequence.pad_sequences(train_words_to_indices, maxlen=max_tweet_length,
                                                                     padding='post', truncating='post')
        test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_words_to_indices, maxlen=max_tweet_length,
                                                                    padding='post', truncating='post')

        print("Shape of the train data: ", train_padded.shape)
        print("Shape of the test data: ", test_padded.shape)

        # len(tokenizer.word_docs) + 2, because of UNKNOWN and PAD.
        return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 3


class Embeddings:
    def __init__(self, filename):
        self.filename = filename

    def loadGloveModel(self):
        print("Loading Glove Model")
        f = open(self.filename, 'r')
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
        print(len(gloveModel), " words loaded!")
        return gloveModel

    def create_embedding_matrix(self, word2idx, dimension):
        max_words = len(word2idx) + 1
        embedding_matrix = np.zeros((max_words, dimension))
        # Load GloVe embeddings.
        embeddings_data = self.loadGloveModel()
        zeros = 1
        for word, index in word2idx.items():
            embedding_vector = embeddings_data.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            else:
                zeros += 1

        print("Shape of the embedding matrix: ", embedding_matrix.shape)
        print("{} words are not found".format(zeros))

        return embedding_matrix


if __name__ == "__main__":
    dr = DataReader('./dataset/small_train.txt', ["id", "tweet", "subtask_a", "subtask_b", "subtask_c"], 'A')
    data, labels = dr.get_data()
    pr = Preprocess(data, labels)
    series_data, series_label = pr.change_data(pr.data, labels)
    # print(series_data)
    # print(series_label)
    # print(type(series_data))
    # print(type(series_label))
    x_train, x_test, y_train, y_test = train_test_split(series_data, series_label, test_size=0.2)
    print(x_train)
    print(y_train)
    train_data, test_data, vocab_freq, word2idx, vocab_size = pr.prepare_data(x_train, x_test, 20)
    print(word2idx)
    embedding = Embeddings("/Users/wangsiwei/PycharmProjects/nlp_2rd/glove_data/glove.twitter.27B.200d.txt")
    # golve_model = embedding.loadGloveModel()
    # print(golve_model["user"])
    embedding_matrix = embedding.create_embedding_matrix(word2idx, 200)
    print(embedding_matrix)
    print(type(embedding_matrix))
    X = tf.placeholder(tf.int32, [None, 20], name="X_input")
    y = tf.placeholder(tf.int64, [None], name="y_label")
    keep_prob = tf.placeholder_with_default(1.0, shape=())

    # Define the variable that will hold the embedding.
    embeddings = tf.get_variable(name="embeddings", shape=[vocab_size, 200],
                                 initializer=tf.constant_initializer(embedding_matrix), trainable=TRAINABLE_EMBEDDINGS)

    # Find the embeddings.
    x_embedded = tf.nn.embedding_lookup(embeddings, X)
    print("Input shape: ", x_embedded.shape)

    # A dynamic RNN.
    lstm_cells = [tf.keras.layers.LSTMCell(units=LSTM_NEURONS, name='lstm_cell')
                  for layer in range(RNN_LAYERS)]
    cells_drop = [tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob)
                  for cell in lstm_cells]
    multi_cell = tf.keras.layers.StackedRNNCells(cells_drop)

    # A bidirectional RNN is used.
    outputs, states = tf.keras.layers.Bidirectional(multi_cell, multi_cell, inputs=x_embedded, dtype=tf.float32)
    print("RNN forward output shape: ", outputs[0].shape)
    print("RNN backward output shape: ", outputs[1].shape)

    outputs = tf.add(outputs[0][:, -1, :], outputs[1][:, -1, :])
    print("RNN squeezed output shape: ", outputs.shape)

    # A hidden layer.
    hidden1 = tf.layers.dense(outputs,
                              NEURONS_HIDDEN_LAYER_1, name="hidden_1", activation='relu')
    print("Hidden layer shape: ", hidden1.shape)

    # A classification layer.
    logits = tf.layers.dense(hidden1, NEURONS_SOFTMAX, name="softmax", activation='softmax')
    print("Logits shape: ", logits.shape)

    # Loss and optimizer.
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

    l2 = LAMBDA_L2_REG * sum([
        tf.nn.l2_loss(tf_var)
        for tf_var in tf.trainable_variables()
        if ("bias" not in tf_var.name or "carry_b" not in tf_var.name)]
    )

    loss += l2
    print("L2 regularized loss: ", loss)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)
    training_op = optimizer.minimize(loss)

    # Predictions and accuracy.
    predictions = tf.argmax(logits, 1, name="predictions")
    correct_predictions = tf.equal(predictions, y)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

    # Initializer.
    init = tf.global_variables_initializer()

    # Saver.
    saver = tf.train.Saver()

    # Summary information for saving.
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    acc_summary = tf.summary.scalar('Accuracy', accuracy)
    summary_op = tf.summary.merge_all()
    logdir = "{}/run-{}/".format(LOG_DIR, now)

    with tf.Session() as sess:
        summary_writer_train = tf.summary.FileWriter(logdir + '/train', tf.get_default_graph())
        summary_writer_test = tf.summary.FileWriter(logdir + '/test')

        init.run()
        for epoch in range(1, EPOCHS + 1):
            X_batch = None
            y_batch = None
            for X_batch, y_batch in shuffle_batch(train_data, y_train, BATCH_SIZE):
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, keep_prob: DROPOUT_KEEP_PROBABILITY})

            # Accuracies after one epoch.
            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: test_data, y: np.reshape(y_test, (y_test.shape[0],))})

            # Get predictions for the test set.
            _, y_pred = sess.run([accuracy, predictions],
                                 feed_dict={X: test_data, y: np.reshape(y_test, (y_test.shape[0],))})

            # Write summaries.
            summary_train_acc = acc_summary.eval(feed_dict={X: train_data, y: np.reshape(y_train, (y_train.shape[0],))})
            summary_test_acc = acc_summary.eval(feed_dict={X: test_data, y: np.reshape(y_test, (y_test.shape[0],))})
            summary_writer_train.add_summary(summary_train_acc, epoch)
            summary_writer_test.add_summary(summary_test_acc, epoch)

            print("Epoch: {} Last batch accuracy: {} Test accuracy: {} F1-Score: {}".format(epoch, acc_train, acc_test,
                                                                                            sklearn_f1_score(y_test,
                                                                                                             y_pred,
                                                                                                             average='macro')))

        saver.save(sess, LOG_DIR + "/tf_model")
    # data = read_from_tsv("/Users/wangsiwei/PycharmProjects/pythonProject2/dataset/small_train.txt",
    #                      ["id", "tweet", "subtask_a", "subtask_b", "subtask_c"])
    # for i, element in enumerate(data):
    #     print(i, element['tweet'])

    # glovemodel = loadGloveModel("/Users/wangsiwei/PycharmProjects/nlp_2rd/glove_data/glove.twitter.27B.200d.txt")
    # print(glovemodel["user"])
