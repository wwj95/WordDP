import os
from PWWS.read_files import split_imdb_files, split_yahoo_files, split_agnews_files
from PWWS.word_level_process import word_process, get_tokenizer, text_to_vector, text_to_vector_for_all
from PWWS.char_level_process import char_process
from PWWS.neural_networks import word_cnn, char_cnn, bd_lstm, lstm
import keras
import spacy
from keras import backend as K
import tensorflow as tf
import argparse
import numpy as np
from PWWS.config import config
import random
from itertools import permutations
from sklearn.utils import shuffle
from wordDP_v4_2 import  abs_diff, exponential_mech, neighbour_sentense_set
from PWWS.adversarial_tools import ForwardGradWrapper

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

K.set_session(tf.Session(config=tf_config))
# GPU config
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(0)
#

parser = argparse.ArgumentParser(
    description='Train a text classifier.')
parser.add_argument('-m', '--model',
                    help='The model of text classifier',
                    choices=['word_cnn', 'char_cnn', 'word_lstm', 'word_bdlstm'],
                    default='word_cnn')
parser.add_argument('-d', '--dataset',
                    help='Data set',
                    choices=['imdb', 'agnews', 'yahoo'],
                    default='imdb')
parser.add_argument('-l', '--level',
                    help='The level of process dataset',
                    choices=['word', 'char'],
                    default='word')
parser.add_argument('-L',
                    help='The L bound',
                    type=int,
                    default = 1)
parser.add_argument('-alpha', 
                    type=int,
                    default = 5)
parser.add_argument('-eps', 
                    type=float,
                    default = 1)

nlp = spacy.load('en_core_web_sm')

def data_preparer(train_texts, y_train, pretrained_model, batch_size):
    train_texts, y_train = shuffle(train_texts, y_train, random_state=0)
    x_batch = train_texts[:batch_size]
    y_batch = y_train[:batch_size]
    x_new = np.zeros_like(x_batch)
    for (i, x) in enumerate(x_batch):
        doc = nlp(x)
        neighbour = neighbour_sentense_set(doc, args.L, int(len(x)/10)+1)
        prob = exponential_mech(x, neighbour, args.aplha, utility_func, args.eps, grad_guide, args.L)
        simu_prob = random.random() 
        cumulate_prob = 0
        for j in range(len(prob)):
            cumulate_prob += prob[j]
            if(simu_prob<cumulate_prob): break
        x_new[i] = neighbour[j]
    x_new = text_to_vector_for_all(x_new, tokenizer, args.dataset)
    return x_new, y_batch


def train_text_classifier():
    dataset = args.dataset
    x_train = y_train = x_test = y_test = None
    if dataset == 'imdb':
        train_texts, train_labels, test_texts, test_labels = split_imdb_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'agnews':
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'yahoo':
        train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    #train_texts, x_train, y_train = shuffle(train_texts, x_train, y_train, random_state=0)

    # Take a look at the shapes
    print('dataset:', dataset, '; model:', args.model, '; level:', args.level)
    print('X_train:', x_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', x_test.shape)
    print('y_test:', y_test.shape)

    log_dir = r'./logs/{}/{}/{}'.format(dataset, "dp_trained", args.model)
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)

    model_path = r'./runs/{}/{}/{}.dat'.format(dataset, "dp_trained", args.model)
    model = batch_size = epochs = None
    assert args.model[:4] == args.level

    if args.model == "word_cnn":
        model = word_cnn(dataset)
        pretrained_model = word_cnn(dataset)
        pretrained_model_path = r'./runs/{}/pretrained_{}.dat'.format(dataset, "word_cnn")
        pretrained_model.load_weights(pretrained_model_path)
        batch_size = config.wordCNN_batch_size[dataset]
        epochs = config.wordCNN_epochs[dataset]
    elif args.model == "word_bdlstm":
        model = bd_lstm(dataset)
        pretrained_model = bd_lstm(dataset)
        pretrained_model_path = r'./runs/{}/pretrained_{}.dat'.format(dataset, "word_bdlstm")
        pretrained_model.load_weights(pretrained_model_path)
        batch_size = config.wordCNN_batch_size[dataset]
        epochs = config.bdLSTM_epochs[dataset]
    elif args.model == "char_cnn":
        model = char_cnn(dataset)
        pretrained_model = char_cnn(dataset)
        pretrained_model_path = r'./runs/{}/pretrained_{}.dat'.format(dataset, "char_cnn")
        pretrained_model.load_weights(pretrained_model_path)
        batch_size = config.wordCNN_batch_size[dataset]
        epochs = config.charCNN_epochs[dataset]
    elif args.model == "word_lstm":
        model = lstm(dataset)
        pretrained_model = lstm(dataset)
        pretrained_model_path = r'./runs/{}/{}.dat'.format(dataset, "word_lstm")
        pretrained_model.load_weights(pretrained_model_path)
        batch_size = config.LSTM_batch_size[dataset]
        epochs = config.LSTM_epochs[dataset]

    print('Train...')
    print('batch_size: ', batch_size, "; epochs: ", epochs)
    global grad_guide
    grad_guide = ForwardGradWrapper(pretrained_model)
    global tokenizer 
    tokenizer = get_tokenizer(args.dataset)
    total = x_train.shape[0]
    for epoch in range(epochs):
        processed = 0
        while processed<total:
            x_batch, y_batch = data_preparer(train_texts, y_train, pretrained_model, batch_size)
            model.fit(x_batch, y_batch,
                    batch_size=batch_size,
                    validation_split = 0,
                    epochs=1)
            processed+=batch_size 
        scores = model.evaluate(x_test, y_test)
        print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
        print('Saving model weights...')
        model.save_weights(model_path)

def utility_func(ori_text, neighbours, alpha, grad_guide, tokenizer):
    ori_prob = grad_guide.predict_prob(text_to_vector(ori_text, tokenizer, args.dataset))
    new_prob = grad_guide.predict_prob(text_to_vector_for_all(neighbours, tokenizer, args.dataset))
    return abs_diff(ori_prob, new_prob, alpha), new_prob

def data_preparer_v2(train_texts, y_train, pretrained_model):
    if not os.path.exists("./certified_eps/train_sample_prob_mapping_{}_{}_{}".format(args.dataset, args.model, args.L)):
        os.mkdir("./certified_eps/train_sample_prob_mapping_{}_{}_{}".format(args.dataset, args.model, args.L))
    for (i, x) in enumerate(train_texts):
        #if i!=27728:
        #    continue
        doc = nlp(x)
        try:
            neighbour = neighbour_sentense_set(doc, args.L, 100)
            if len(neighbour)==1:
                neighbour.append(neighbour[0])
            prob=exponential_mech(x, neighbour, args.alpha, utility_func, float(args.eps), grad_guide, tokenizer, args.L, True)
        except:
            import pdb; pdb.set_trace()
        np.save("./certified_eps/train_sample_prob_mapping_{}_{}_{}/{}".format(args.dataset, args.model, args.L, i), [neighbour, prob])
        print(i)


def read_noised_data(train_texts):
    x_new = []
    for(i, train_x) in enumerate(train_texts):
        cand = np.load("./certified_eps/train_sample_prob_mapping_{}_{}_{}/{}.npy".format(args.dataset, args.model,args.L, i), allow_pickle = True)
        prob = cand[1]
        neighbour = cand[0]
        simu_prob = random.random() 
        cumulate_prob = 0
        for j in range(len(prob)):
            cumulate_prob += float(prob[j])
            if(simu_prob<cumulate_prob): break
        try:
            x_new.append(neighbour[j])
        except:
            import pdb; pdb.set_trace()
    x_train =  text_to_vector_for_all(x_new, tokenizer, args.dataset)
    return x_train

def train_text_classifier_v2():
    dataset = args.dataset
    x_train = y_train = x_test = y_test = None
    if dataset == 'imdb':
        train_texts, train_labels, test_texts, test_labels = split_imdb_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'agnews':
        train_texts, train_labels, test_texts, test_labels = split_agnews_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    elif dataset == 'yahoo':
        train_texts, train_labels, test_texts, test_labels = split_yahoo_files()
        if args.level == 'word':
            x_train, y_train, x_test, y_test = word_process(train_texts, train_labels, test_texts, test_labels, dataset)
        elif args.level == 'char':
            x_train, y_train, x_test, y_test = char_process(train_texts, train_labels, test_texts, test_labels, dataset)
    #train_texts, x_train, y_train = shuffle(train_texts, x_train, y_train, random_state=0)

    # Take a look at the shapes
    print('dataset:', dataset, '; model:', args.model, '; level:', args.level)
    print('X_train:', x_train.shape)
    print('y_train:', y_train.shape)
    print('X_test:', x_test.shape)
    print('y_test:', y_test.shape)

    log_dir = r'./logs/{}/{}/{}'.format(dataset, "dp_trained", args.model)
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)

    model_path = r'./runs/{}/{}/{}.dat'.format(dataset, "dp_trained", args.model)
    model = batch_size = epochs = None
    assert args.model[:4] == args.level

    if args.model == "word_cnn":
        model = word_cnn(dataset)
        pretrained_model = word_cnn(dataset)
        pretrained_model_path = r'./runs/{}/pretrained_{}.dat'.format(dataset, "word_cnn")
        pretrained_model.load_weights(pretrained_model_path)
        batch_size = config.wordCNN_batch_size[dataset]
        epochs = config.wordCNN_epochs[dataset]
    elif args.model == "word_bdlstm":
        model = bd_lstm(dataset)
        pretrained_model = bd_lstm(dataset)
        pretrained_model_path = r'./runs/{}/pretrained_{}.dat'.format(dataset, "word_bdlstm")
        pretrained_model.load_weights(pretrained_model_path)
        batch_size = config.wordCNN_batch_size[dataset]
        epochs = config.bdLSTM_epochs[dataset]
    elif args.model == "char_cnn":
        model = char_cnn(dataset)
        pretrained_model = char_cnn(dataset)
        pretrained_model_path = r'./runs/{}/pretrained_{}.dat'.format(dataset, "char_cnn")
        pretrained_model.load_weights(pretrained_model_path)
        batch_size = config.wordCNN_batch_size[dataset]
        epochs = config.charCNN_epochs[dataset]
    elif args.model == "word_lstm":
        model = lstm(dataset)
        pretrained_model = lstm(dataset)
        pretrained_model_path = r'./runs/{}/{}.dat'.format(dataset, "word_lstm")
        pretrained_model.load_weights(pretrained_model_path)
        batch_size = config.LSTM_batch_size[dataset]
        epochs = config.LSTM_epochs[dataset]

    print('Train...')
    print('batch_size: ', batch_size, "; epochs: ", epochs)
    global grad_guide
    grad_guide = ForwardGradWrapper(pretrained_model)
    global tokenizer 
    tokenizer = get_tokenizer(args.dataset)
    total = x_train.shape[0]
    #if not os.path.exists("./certified_eps/train_sample_prob_mapping_{}_{}_{}".format(dataset, args.model,args.L)):
    data_preparer_v2(train_texts, y_train, pretrained_model)
    for epoch in range(epochs):
        x_train = read_noised_data(train_texts)
        model.fit(x_train, y_train,
                batch_size=batch_size,
                validation_split = 0,
                epochs=1)
        
    scores = model.evaluate(x_test, y_test)    
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    print('Saving model weights...')
    model.save_weights(model_path)


if __name__ == '__main__':
    args = parser.parse_args()
    train_text_classifier_v2()
