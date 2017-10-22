import tensorflow as tf
import numpy as np
import glob  # this will be useful when reading reviews from file
import os
import tarfile
import string

# TODO add multilayer
stopwords_set = {'a', 'about', 'above', 'across', 'after', 'again', 'against', 'all', 'almost', 'alone', 'along',
                 'already', 'also', 'although', 'always', \
                 'among', 'an', 'and', 'another', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area',
                 'areas', 'around', 'as', 'ask', 'asked', 'asking', \
                 'asks', 'at', 'away', 'b', 'back', 'backed', 'backing', 'backs', 'be', 'became', 'because', 'become',
                 'becomes', 'been', 'before', 'began', 'behind', \
                 'being', 'beings', 'best', 'better', 'between', 'big', 'both', 'but', 'by', 'c', 'came', 'can',
                 'cannot', 'case', 'cases', 'certain', 'certainly', \
                 'clear', 'clearly', 'come', 'could', 'd', 'did', 'differ', 'different', 'differently', 'do', 'does',
                 'done', 'down', 'down', 'downed', 'downing', \
                 'downs', 'during', 'e', 'each', 'early', 'either', 'end', 'ended', 'ending', 'ends', 'enough', 'even',
                 'evenly', 'ever', 'every', 'everybody', \
                 'everyone', 'everything', 'everywhere', 'f', 'face', 'faces', 'fact', 'facts', 'far', 'felt', 'few',
                 'find', 'finds', 'first', 'for', 'four', 'from', \
                 'full', 'fully', 'further', 'furthered', 'furthering', 'furthers', 'g', 'gave', 'general', 'generally',
                 'get', 'gets', 'give', 'given', 'gives', 'go', \
                 'going', 'good', 'goods', 'got', 'great', 'greater', 'greatest', 'group', 'grouped', 'grouping',
                 'groups', 'h', 'had', 'has', 'have', 'having', 'he', \
                 'her', 'here', 'herself', 'high', 'high', 'high', 'higher', 'highest', 'him', 'himself', 'his', 'how',
                 'however', 'i', 'if', 'important', 'in', \
                 'interest', 'interested', 'interesting', 'interests', 'into', 'is', 'it', 'its', 'itself', 'j', 'just',
                 'k', 'keep', 'keeps', 'kind', 'knew', 'know', \
                 'known', 'knows', 'l', 'large', 'largely', 'last', 'later', 'latest', 'least', 'less', 'let', 'lets',
                 'like', 'likely', 'long', 'longer', 'longest', 'm', \
                 'made', 'make', 'making', 'man', 'many', 'may', 'me', 'member', 'members', 'men', 'might', 'more',
                 'most', 'mostly', 'mr', 'mrs', 'much', 'must', 'my', \
                 'myself', 'n', 'necessary', 'need', 'needed', 'needing', 'needs', 'never', 'new', 'newer', 'newest',
                 'next', 'nobody', 'noone', 'nothing', 'now', 'nowhere' ,\
                 'o', 'of', 'off', 'often', 'old', 'older', 'oldest', 'on', 'once', 'one', 'only', 'open', 'opened', 'opening',
                 'opens', 'or', 'other', 'others', \
                 'our', 'out', 'over', 'p', 'part', 'parted', 'parting', 'parts', 'per', 'perhaps', 'possible',
                 'present', 'presented', 'presenting', 'problem', 'put', \
                 'puts', 'q', 'quite', 'r', 'rather', 'really', 'right', 'right', 's', 'said', 'same', 'saw', 'say',
                 'says', 'see', 'seem', 'seemed', 'seeming', 'seems', \
                 'sees', 'several', 'shall', 'she', 'should', 'show', 'showed', 'showing', 'shows', 'since', 'small',
                 'smaller', 'smallest', 'so', 'some', 'somebody', \
                 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'still', 'such', 'sure', 't', 'take',
                 'taken', 'than', 'that', 'the', 'their', \
                 'them', 'then', 'there', 'therefore', 'these', 'they', 'this', 'those', 'though', 'three', 'through',
                 'thus', 'to', 'today', 'together', 'too', 'took', \
                 'toward', 'turn', 'turned', 'turning', 'turns', 'two', 'u', 'under', 'until', 'up', 'upon', 'us',
                 'use', 'used', 'uses', 'v', 'very', 'w', 'wanting', \
                 'was', 'way', 'ways', 'we', 'well', 'wells', 'went', 'were', 'what', 'when', 'where', 'whether',
                 'which', 'while', 'who', 'whole', 'whose', 'why', \
                 'will', 'with', 'within', 'without', 'work', 'worked', 'working', 'works', 'would', 'x', 'y', 'year',
                 'years', 'yet', 'you', 'young', 'younger', 'youngest', 'your', 'yours', 'z'}

## tunable, have a test on this
batch_size = 64


def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    dir = None
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data2/')):
        with tarfile.open('reviews.tar.gz', "r") as tarball:
            dir = os.path.dirname(__file__)
            tarball.extractall(os.path.join(dir, 'data2/'))
    else:
        dir = os.path.dirname(__file__)

    file_list = glob.glob(os.path.join(dir, 'data2/pos/*'))
    file_list.extend(glob.glob(os.path.join(dir, 'data2/neg/*')))

    output = []

    for f in file_list:
        with open(f, "r") as openf:
            s = openf.read()
            no_punct = ''.join(c for c in s if c not in string.punctuation)
            words = no_punct.split()

            word_arr = []
            word_arr_index = 0
            for word in words:
                if word_arr_index >= 40:
                    break
                word = word.lower()
                if word in stopwords_set:
                    # word_arr_index do not need to increase 1 since 
                    # nothing added to word_arr
                    continue
                else:
                    # print(word, end=" ")
                    if word in glove_dict:
                        # print("true")
                        word_arr.append(glove_dict[word])
                    else:
                        # print("false")
                        word_arr.append(0)
                    word_arr_index += 1

            if word_arr_index < 40:
                for i in range(word_arr_index, 40):
                    word_arr.append(0)

            # add vectorized review to output list
            output.append(word_arr)

    # print(output[0])
    # print(output[10])
    data = np.array(output)
    # print('end load_data')
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    data = open("glove.6B.50d.txt", 'r', encoding="utf-8")
    # if you are running on the CSE machines, you can load the glove data from here
    # data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")

    embeddings = []
    word_index_dict = {'UNK': 0}
    index = 1

    for entry in data:
        entry_arr = entry.strip().split(' ')
        # entry_arr[0] is the word, the following is the word2vec value
        word_index_dict[entry_arr[0]] = index
        index += 1
        temp_arr = [float(i) for i in entry_arr[1:]]
        embeddings.append(temp_arr)

    # use 40 0s to represent `UNK`
    embeddings.insert(0, [0.0] * len(embeddings[0]))
    # embeddings = np.array(embeddings)
    # print('embeddings: ', embeddings)
    # print(len(embeddings[0]))
    # print('end load_glove_embeddings')
    return embeddings, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""
    dropout_keep_prob = tf.placeholder_with_default(0.9, shape=())

    num_words_per_review = 40

    input_data = tf.placeholder(tf.int32, [batch_size, num_words_per_review], name="input_data")
    labels = tf.placeholder(tf.int32, [batch_size, 2], name="labels")

    num_input = len(glove_embeddings_arr[0])  # data input (shape: 40 * 50)
    timesteps = num_words_per_review  # timesteps, 40 words, and each words represent by 50 numbers

    # this is tunable
    num_hidden = 64
    # there will only be two classes
    num_classes = 2
    learning_rate = 0.001
    # num_layers = 3
    # dropout_rate = 0.1

    with tf.name_scope("weight_bias"):
        # the `2 *` is used for backward and forward rnn, if only backward rnn used
        # we do not need the `2 *`
        weights =  tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))
        biases =   tf.Variable(tf.random_normal([num_classes]))


    # input is batch_size * 40
    # what we will transform to is batch_size * 40 * 50(50 is how many elements in word2vec)
    # 2d embedding array and 1d ids
    with tf.name_scope("input"):
        x = tf.nn.embedding_lookup(
            tf.constant(glove_embeddings_arr),
            tf.reshape(input_data, [-1]))
        x = tf.reshape(x, [batch_size, timesteps, num_input])
        x = tf.unstack(x, timesteps, 1)

    # cells = []
    # for _ in range(num_layers):
    #   cell = tf.contrib.rnn.BasicLSTMCell(num_hidden)
    #   cell = tf.contrib.rnn.DropoutWrapper(
    #       cell, output_keep_prob=1.0 - dropout_rate)
    #   cells.append(cell)
    # cell = tf.contrib.rnn.MultiRNNCell(cells)
    # lstm_cell = cell # useless replace name later

    with tf.name_scope("forward_unit_rnn"):
        fw = tf.contrib.rnn.BasicLSTMCell(num_hidden ,forget_bias=1.0)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(
            fw, output_keep_prob=dropout_keep_prob)

    with tf.name_scope("backward_unit_rnn"):
        bw = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(
            bw, output_keep_prob=dropout_keep_prob)

    # lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers)
    # output of static_rnn is a [batch_size, n_hidden] tensor list
    # outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
    with tf.name_scope("bidirectional_rnn"):
        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                            dtype=tf.float32)

    with tf.name_scope("logits"):
        # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
        logits = tf.matmul(outputs[-1], weights) + biases

    # with tf.name_scope("loss"):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels), name='loss')

    with tf.name_scope("train"):
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)

    # accuray
    with tf.name_scope("accuracy"):
        prediction = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
