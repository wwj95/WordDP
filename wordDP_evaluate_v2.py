from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
# GPU config
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="args.GPU"
#
import re
from PWWS.evaluate_fool_results import read_adversarial_file
from PWWS.paraphrase import _generate_synonym_candidates
from PWWS.read_files import split_imdb_files, split_yahoo_files, split_agnews_files
from PWWS.word_level_process import word_process, get_tokenizer, text_to_vector, text_to_vector_for_all
from PWWS.char_level_process import char_process, doc_process_for_all, get_embedding_dict
from PWWS.neural_networks import word_cnn, char_cnn, bd_lstm, lstm
import spacy
import tensorflow as tf
from keras import backend as K
from PWWS.adversarial_tools import ForwardGradWrapper
import numpy as np
import copy
import random
from itertools import permutations
from PWWS.wordDP_v4_2 import neighbour_sentense_set, wordDP_defense, abs_diff


parser = argparse.ArgumentParser(
    description='Craft adversarial examples for a text classifier.')
parser.add_argument('--start',
                    help='Amount of clean(test) samples to fool',
                    type=int, default=11500)
parser.add_argument('--clean_samples_cap',
                    help='Amount of clean(test) samples to fool',
                    type=int, default=2000)
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
parser.add_argument('-ita',
                    type=int,
                    default = 2)
parser.add_argument('-L',
                    type=int)    
parser.add_argument('-eps',
                    type=float)  
parser.add_argument('-b',
                    help='The bound of the candidate set size',
                    type=int,
                    default = 200)          


nlp = spacy.load('en_core_web_sm')

def utility_func(ori_text, neighbours, alpha, grad_guide, tokenizer):
    ori_prob = grad_guide.predict_prob(text_to_vector(ori_text, tokenizer, args.dataset))
    new_prob = grad_guide.predict_prob(text_to_vector_for_all(neighbours, tokenizer, args.dataset))
    return abs_diff(ori_prob, new_prob, alpha)

def main():
    config = tf.ConfigProto(allow_soft_placement=True) 
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    dataset = args.dataset
    global tokenizer
    tokenizer = get_tokenizer(dataset)

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

    if args.model == "word_cnn":
        model = word_cnn(dataset)
        model_path = r'./runs/{}/pretrained_{}.dat'.format(dataset, "word_cnn")
        model.load_weights(model_path)

    elif args.model == "word_bdlstm":
        model = bd_lstm(dataset)
        model_path = r'./runs/{}/{}.dat'.format(dataset, "word_bdlstm")
        model.load_weights(model_path)

    elif args.model == "char_cnn":
        model = char_cnn(dataset)
        model_path = r'./runs/{}/pretrained_{}.dat'.format(dataset, "char_cnn")
        model.load_weights(model_path)

    elif args.model == "word_lstm":
        model = lstm(dataset)
        model_path = r'./runs/{}/{}.dat'.format(dataset, "word_lstm")
        model.load_weights(model_path)

    print('model path:', model_path)
    grad_guide = ForwardGradWrapper(model)
    adv_text_path = r'./fool_result/{}/{}/adv_{}.txt'.format(dataset, args.model, args.clean_samples_cap)
    print('adversarial file:', adv_text_path)
    clean_text = test_texts[args.start:args.start+args.clean_samples_cap]
    adv_text = read_adversarial_file(adv_text_path)
    count = 0
    for i in range(len(adv_text)):
        doc = nlp(adv_text[i])
        neighbour = neighbour_sentense_set(doc, args.L, args.b)
        res = wordDP_defense(adv_text[i], neighbour, model, utility_func, 200, args.eps, tokenizer) 
        adv_label = np.argmax(res)
        print (adv_label)
        if(adv_label == 1 and i<1000): count+=1
        if(adv_label == 0 and i>=1000): count+=1
    print (count)


if __name__ == '__main__':
    args = parser.parse_args()
    main()








