from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# GPU config
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(1)
#
import argparse
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
import time
from collections import OrderedDict

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
parser.add_argument('-alpha',
                    type=int, 
                    default = 5) 
parser.add_argument('-b',
                    help='The bound of the candidate set size',
                    type=int,
                    default = 100) 
parser.add_argument('-clean',
                    type=bool)   

nlp = spacy.load('en_core_web_sm')

def generate_pert_set(doc, size):
    pert_set = []

    for i in range(size):
        #import pdb; pdb.set_trace()
        sen = []
        for word_index, word in enumerate(doc):
            
            sys = _generate_synonym_candidates(token = word, token_position = word_index)
            if len(sys)>0:
                sen.append(sys[np.random.randint(0, len(sys))].candidate_word)
            else:
                sen.append(word.text)
        final_sen = " ".join(sen)
        pert_set.append(final_sen)
    
    return pert_set

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def qx_overlapping(word, word_index):
    sys = _generate_synonym_candidates(token = word, token_position = word_index)
    overlap = 0
    if len(sys)>0:
        sys2 = [sys[i].candidate_word for i in range(len(sys))]
        for s in sys2:
            #import pdb; pdb.set_trace()
            q_s = _generate_synonym_candidates(token = nlp(s)[0], token_position = word_index)
            if len(q_s)>0:
                q_s2 = [q_s[i].candidate_word for i in range(len(q_s))]
                if overlap!=0:
                    if len(intersection(sys2, q_s2))/len(sys) != 0:
                        #import pdb; pdb.set_trace()
                        overlap = min(len(intersection(sys2, q_s2))/len(sys), overlap)
                    
                else:
                    overlap = len(intersection(sys2, q_s2))/len(sys)
    return overlap

def safer_certified(doc, y, yb):
    #import pdb; pdb.set_trace()
    qx_dic = np.zeros(len(doc))
    for word_index, word in enumerate(doc):
        qx = qx_overlapping(word, word_index)
        qx_dic[word_index] = qx
    sorted_qx = np.sort(qx_dic)
    #import pdb; pdb.set_trace()
    return y - yb - 2*(1 - np.prod(sorted_qx[-10:-1]))>0


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
    clean_label = y_test[args.start:args.start+args.clean_samples_cap]


    adv_text = read_adversarial_file(adv_text_path)
    x_adv = text_to_vector_for_all(adv_text, tokenizer, args.dataset)
    score_adv = model.evaluate(x_adv, y_test[args.start:args.start+args.clean_samples_cap])
    print('adv test_loss: %f, accuracy: %f' % (score_adv[0], score_adv[1]))


    certified = 0
    count = 0
    start = time.time()
    for i in range(len(clean_text)):
        
        sentence = clean_text[i]
        doc = nlp(sentence)
        pert_set = generate_pert_set(doc,100)
        #import pdb; pdb.set_trace()
        sets = text_to_vector_for_all(pert_set, tokenizer, args.dataset)
        pred = grad_guide.predict_prob(input_vector=sets)
        res = np.mean(pred,0)
        label = np.argmax(res) 
        target_res = res[np.argmax(clean_label[i])]
        res[np.argmax(clean_label[i])] = -1
        largest_exc_tgt = np.argmax(res)
        if safer_certified(doc, target_res, res[largest_exc_tgt]):
            certified+=1
            
            adv_doc = nlp(adv_text[i])
            adv_pert_set = generate_pert_set(adv_doc,10)
            sets = text_to_vector_for_all(adv_pert_set, tokenizer, args.dataset)
            pred = grad_guide.predict_prob(input_vector=sets)
            res = np.mean(pred,0)
            adv_label = np.argmax(res)
            if adv_label == np.argmax(clean_label[i]):
                count+=1
            print("Successed Certified of sample:", i, "Arg prediction:", adv_label, "Original Label",np.argmax(clean_label[i]), "Certified:", certified, "Counts:", count)

        else:
            print("Failed Certified of sample:", i,  "Certified:", certified, "Counts:", count)   
        if i%100==0:
            curr = time.time()
            print(curr-start)
    print(certified, count)

if __name__ == '__main__':
    args = parser.parse_args()
    main()
