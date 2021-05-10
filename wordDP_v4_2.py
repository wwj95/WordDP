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
import time
import random
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
parser.add_argument('-L_adv',
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

def get_index_ori(change_tuple_file, L):
    result = []
    for idx in range(len(change_tuple_file)-1):
        data = change_tuple_file[idx].split("[(")[1].split('), (')
        if len(data)<=L:
            result.append(idx)
    return result

def get_index(change_tuple_file, L):
    result = []
    for idx in range(len(change_tuple_file)-1):
        res = change_tuple_file[idx].split("[(")
        data = res[1].split('), (')
        index = int(res[0])
        if len(data)<=L:
            result.append(index)
    return result

def word_sub(doc):
    ori_doc = []
    for i in range(len(doc)):
        ori_doc.append(doc[i].text)
    return " ".join(ori_doc)


def neighbour_sentense_set(doc, L, size):
    cand_dict = []
    cand_len_list = []
    for word_index, word in enumerate(doc):
        cand_set_build = _generate_synonym_candidates(token = word, token_position = word_index)
        cand_set = [i.candidate_word for i in cand_set_build]
        cand_dict.append(cand_set)
        cand_len_list.append(len(cand_set))

    prob = [i/sum(cand_len_list) for i in cand_len_list]

    neighbour_set = []
    #import pdb; pdb.set_trace()
    while(len(neighbour_set)<size):
        loc = []
        for i in range(L):
            simu_prob = random.random()
            cumulate_prob = 0
            for j in range(len(prob)):
                cumulate_prob += prob[j]
                if(simu_prob<cumulate_prob): break
            loc.append(j)
        sen = []
        for word_index, word in enumerate(doc):
            if(word_index not in loc):
                sen.append(word.text)
                continue
            sys = cand_dict[word_index]
            sen.append(sys[np.random.randint(0, len(sys))])

        final_sen = " ".join(sen)
        neighbour_set.append(final_sen)

    return list(set(neighbour_set))


def neighbour_sentense_set_v1(doc, L, size):
    cand_dict = []
    cand_len_list = []
    for word_index, word in enumerate(doc):
        cand_set_build = _generate_synonym_candidates(token = word, token_position = word_index)
        #import pdb; pdb.set_trace()
        cand_set = [i.candidate_word for i in cand_set_build]
        cand_dict.append(cand_set)
        cand_len_list.append(len(cand_set))

    prob = [i/sum(cand_len_list) for i in cand_len_list]

    neighbour_set = []
    #import pdb; pdb.set_trace()
    while(len(neighbour_set)<size):
        loc = []
        for i in range(L):
            simu_prob = random.random()
            cumulate_prob = 0
            for j in range(len(prob)):
                cumulate_prob += prob[j]
                if(simu_prob<cumulate_prob): break
            loc.append(j)
        sen = []
        for word_index, word in enumerate(doc):
            if(word_index not in loc):
                sen.append(word.text)
                continue
            sys = cand_dict[word_index]
            sen.append(sys[np.random.randint(0, len(sys))])

        final_sen = " ".join(sen)
        neighbour_set.append(final_sen)

    return neighbour_set



def abs_diff(tar_arr, cand_arr, alpha):
    #import pdb; pdb.set_trace()
    return np.exp(-(np.max(abs(tar_arr-cand_arr),axis = 1)))


def abs_diff_v1(tar_arr, cand_arr, alpha):
    #diff = np.linalg.norm(tar_arr-cand_arr, ord=2, axis = 1)
    diff = np.max(abs(tar_arr-cand_arr),axis = 1)
    #mean = np.mean(diff)
    #std = np.std(diff)
    #diff = (diff-mean)/std
    #import scipy.stats
    #score = scipy.stats.norm(loc=0, scale=1).pdf(diff)
    #score = scipy.stats.norm(loc=mean, scale=std).pdf(diff)
    #import pdb; pdb.set_trace()

    return -np.log(diff)


def utility_func(ori_text, neighbours, alpha, grad_guide, tokenizer):
    ori_prob = grad_guide.predict_prob(text_to_vector(ori_text, tokenizer, args.dataset))
    new_prob = grad_guide.predict_prob(text_to_vector_for_all(neighbours, tokenizer, args.dataset))
    #import pdb; pdb.set_trace()
    return abs_diff(ori_prob, new_prob, alpha), new_prob

def exponential_mech(ori_text, neighbours, alpha, utility_func, eps, grad_guide, tokenizer, L, prob_only=False):
    utility_score, pred = utility_func(ori_text, neighbours, alpha, grad_guide, tokenizer)
    #print(utility_score)
    #u_sensitivity = np.max(utility_score)
    u_sensitivity = 1 - np.exp(-1)
    prob_score = np.exp(eps*utility_score/(2*u_sensitivity))
    prob = prob_score/np.sum(prob_score)
    #import pdb; pdb.set_trace()
    if prob_only:
        return prob
    return pred, prob

def cond_check(expexted_1, expected_2, lamda, k, N, eps):
    sub = np.sqrt(1/(2*N)*np.log(2*k/(1-lamda)))
    return (expexted_1-sub)>=np.exp(2*eps)*(expected_2+sub)
   
def wordDP_defense(ori_text, neighbour, model, utility_func, N, eps, tokenizer):
    grad_guide = ForwardGradWrapper(model)
    pred, prob = exponential_mech(ori_text, neighbour, args.alpha, utility_func,eps, grad_guide, tokenizer, args.L)
    #sets = text_to_vector_for_all(neighbour, tokenizer, args.dataset)
    #pred = grad_guide.predict_prob(input_vector=sets)
    agr_pred  = 0
    for i in range(len(pred)):
        agr_pred+=pred[i]*prob[i]
    return agr_pred


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
        model_path = r'./runs/{}/dp_trained/{}.dat'.format(dataset, "word_lstm")
        model.load_weights(model_path)

    print('model path:', model_path)
    grad_guide = ForwardGradWrapper(model)
    adv_text_path = r'./fool_result/{}/{}/adv_{}.txt'.format(dataset, args.model, args.clean_samples_cap)
    print('adversarial file:', adv_text_path)
    change_tuple_path = r'./fool_result/{}/{}/change_tuple_{}.txt'.format(args.dataset, args.model, args.clean_samples_cap)
    change_tuple_file = open(change_tuple_path).read().split("\n")
    '''
    index_L = get_index(change_tuple_file, args.L_adv)
    index_L_2 = [i+args.start for i in index_L]
    clean_text = [test_texts[i] for i in index_L_2]
    clean_label = [y_test[i] for i in index_L_2]
    adv_text = [read_adversarial_file(adv_text_path)[i] for i in index_L]
    '''
    index_L = get_index(change_tuple_file, args.L_adv)
    #import pdb; pdb.set_trace()
    index_L_2 = [i-args.start for i in index_L]
    clean_text = [test_texts[i] for i in index_L]
    clean_label = [y_test[i] for i in index_L]
    adv_text = [read_adversarial_file(adv_text_path)[i] for i in index_L_2]
    
    #import pdb; pdb.set_trace()
    adv = adv_text
    x_adv = text_to_vector_for_all(adv, tokenizer, args.dataset)
    score_adv = model.evaluate(x_adv, np.stack(clean_label, axis=0))
    print('adv test_loss: %f, accuracy: %f' % (score_adv[0], score_adv[1]))

    x_clean = text_to_vector_for_all(clean_text, tokenizer, args.dataset)
    score_clean = model.evaluate(x_clean, np.stack(clean_label, axis=0))
    print('clean test_loss: %f, accuracy: %f' % (score_clean[0], score_clean[1]))

    clean_count = 0
    certified = 0
    count = 0
    both = 0
    start = time.time()
    for i in range(len(clean_text)):        
    #for i in range(2):
        sentence = clean_text[i]
        doc = nlp(sentence)
        ori_text = word_sub(doc)
        neighbour = neighbour_sentense_set(doc, args.L, args.b)
        res = wordDP_defense(ori_text, neighbour, model, utility_func, 2000, args.eps, tokenizer)
        #import pdb; pdb.set_trace()
        """        
        if args.clean:
            label = np.argmax(res) 
            if label==np.argmax(clean_label[i]):
                clean_count+=1
        if i == 200:
            break
        print (i)
    result = open("./result_0327", "a")
    result.write("eps: {}, L: {}, L_adv: {}\n".format(args.eps, args.L, args.L_adv))
    result.write("{}, {}\n".format(i, clean_count))
        #import pdb; pdb.set_trace()
        """
        # prediction label
        pred_label = np.argmax(res)

        target_res = res[np.argmax(clean_label[i])]
        res[np.argmax(clean_label[i])] = -1
        #target_res = res[pred_label]
        #res[pred_label] = -1
        largest_exc_tgt = np.argmax(res)
        check = cond_check(target_res, res[largest_exc_tgt], 0.95, args.ita, args.clean_samples_cap, args.eps)
        
        if(np.argmax(clean_label[i])==1):
            check = cond_check(res[1], res[0], 0.95, args.ita, args.clean_samples_cap, args.eps)
        else:
            check = cond_check(res[0], res[1], 0.95, args.ita, args.clean_samples_cap, args.eps)
        
        
        doc = nlp(adv_text[i])
        neighbour = neighbour_sentense_set(doc, args.L, args.b)
        #grad_guide.predict_prob(text_to_vector(ori_text, tokenizer, args.dataset))
        #grad_guide.predict_prob(text_to_vector(adv_text[i], tokenizer, args.dataset))
        
        #if check:
            #import pdb; pdb.set_trace()
        
        res = wordDP_defense(adv_text[i], neighbour, model, utility_func, 10000, args.eps, tokenizer) 
        adv_label = np.argmax(res)
        #import pdb; pdb.set_trace()
        
        if(adv_label == np.argmax(clean_label[i])): 
            count+=1
        
        if check == True:
            certified+=1
            print("Successed Certified of sample:", i, "Arg prediction:", adv_label, "Original Label",np.argmax(clean_label[i]), "Certified:", certified, "Counts:", count, both, clean_count )
        else:
            print("Failed Certified of sample:", i, "Arg prediction:", adv_label, "Original Label",np.argmax(clean_label[i]), "Certified:", certified, "Counts:", count, both, clean_count )
        
        if check==True and adv_label == np.argmax(clean_label[i]):
            both+=1
        #import pdb; pdb.set_trace()
        if i%100==0:
            curr = time.time()
            print(curr-start)
        if i == 200:
            break
        print (len(clean_text), certified, count, both)
    
    result = open("./result_0327", "a")
    result.write("eps: {}, L: {}, L_adv: {}\n".format(args.eps, args.L, args.L_adv))
    result.write("{}, {}, {}, {}\n".format(i, certified, count, both))
    result.close()
    

if __name__ == '__main__':
    args = parser.parse_args()
    main()

