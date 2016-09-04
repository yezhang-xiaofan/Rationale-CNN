import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
def build_data_cv(data_folder, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_folder = data_folder[0]
    neg_folder = data_folder[1]
    vocab = defaultdict(float)
    max_sen_len = 0
    max_doc_len = 0
    total_sen_len = 0.0
    total_doc_len = 0.0
    num_sen = 0.0
    num_doc = 0.0
    write_file_dir_pos = "Rats_pos_sentence/"
    write_file_dir_neg = "Rats_neg_sentence/"
    num_rational = 0.0
    for pos_file in os.listdir(pos_folder):
        if not pos_file.endswith('txt'):continue
        #if pos_file.split('_')[1][0] == '9': continue   #whether to include the 9th fold
        #print pos_file
        with open(os.path.join(pos_folder,pos_file), "rb") as f:
            write_file = open(write_file_dir_pos+pos_file,"wb")
            cur_doc = []
            num_doc += 1
            line = f.readline()
            #for line in lines:
            '''
            words = line.strip().split()
            for word in words:
                if word != '<POS>' and word != '</POS>':
                    vocab[word] += 1
            start_indices = [i for i, x in enumerate(words) if x == "<POS>"]
            end_indices = [i for i, x in enumerate(words) if x == "</POS>"]
            if len(start_indices) != len(end_indices): print "error found in file " + str(file)
            indices_pair = zip(start_indices,end_indices)
            remove_indices = []
            for i in indices_pair:
                start_index = i[0]
                end_index = i[1]
                a = words[start_index+1:end_index]   #generate psudo examples
                cur_doc.append([' '.join(a).strip(),1])
                num_rational += 1
                write_file.write(" ".join(a).strip()+"\t"+str(1)+"\n")
                remove_indices += range(start_index+1,end_index)   ##remove contents between tags
                num_sen += 1
            remove_tag_sen = []
            for w in words:
                if w == "<POS>" or w == "</POS>": remove_tag_sen.append('.')
                else: remove_tag_sen.append(w)
            remove_tag_sen = [v for i, v in enumerate(remove_tag_sen) if i not in remove_indices]
            remove_tag_sen = ' '.join(remove_tag_sen)
            sentences = sent_tokenize(remove_tag_sen)
            for s in sentences:
                if len(s.split())<=3:continue
                cur_doc.append([s.strip(),0])
                write_file.write(s.strip()+"\t"+str(0)+"\n")
                num_sen += 1
            '''
            sentences = sent_tokenize(line.strip())
            #sentences = [lines]
            for s in sentences:
                    num_sen += 1
                    sentence = s.split()   ##words comprising sentence
                    cur_sen_len = len(sentence)
                    if cur_sen_len > max_sen_len: max_sen_len = cur_sen_len
                    total_sen_len += cur_sen_len
                    #words = set(sentence)
                    for word in sentence:
                        vocab[word] += 1
                    #if  ("POS" in s or "/POS" in s): continue
                    if ("POS" in s or "/POS" in s):
                        s = s.strip()
                        s = s.replace('<POS>','')
                        s = s.replace('</POS>','')
                        s = s.replace('< POS >','')
                        s = s.replace('< /POS >','')
                        write_file.write(s+"\t"+str(1)+"\n")
                        num_rational += 1
                        cur_doc.append([s,1])
                        continue
                    if (cur_sen_len) <= 3: continue
                    cur_doc.append([" ".join(sentence).strip(),0])
                    write_file.write(" ".join(sentence).strip()+"\t"+str(0)+"\n")
            if len(cur_doc) > max_doc_len: max_doc_len = len(cur_doc)
            total_doc_len += len(cur_doc)
            datum  = {"y":1,
                          "text": cur_doc,
                          "split": int(pos_file.split('_')[1][0])}

            revs.append(datum)
            write_file.write("1")
            write_file.close()

    for neg_file in os.listdir(neg_folder):
        if not neg_file.endswith('txt'):continue
        #if neg_file.split('_')[1][0] == '9': continue      #whether to inclue 9th fold
        with open(os.path.join(neg_folder,neg_file), "rb") as f:
            cur_doc = []
            num_doc += 1
            line = f.readline()
            write_file = open(write_file_dir_neg+neg_file,"wb")

            '''
            words = line.strip().split()
            for word in words:
                if word != '<NEG>' and word != '</NEG>':
                    vocab[word] += 1
            start_indices = [i for i, x in enumerate(words) if x == "<NEG>"]
            end_indices = [i for i, x in enumerate(words) if x == "</NEG>"]
            if len(start_indices) != len(end_indices): print "error found in file " + str(file)
            indices_pair = zip(start_indices,end_indices)
            remove_indices = []
            for i in indices_pair:
                start_index = i[0]
                end_index = i[1]
                a = words[start_index+1:end_index]   #generate psudo examples
                cur_doc.append([" ".join(a).strip(),1])
                num_rational += 1
                write_file.write(" ".join(a).strip()+"\t"+str(1)+"\n")
                remove_indices += range(start_index+1,end_index)
                num_sen += 1
            remove_tag_sen = []
            for w in words:
                if w == "<NEG>" or w == "</NEG>": remove_tag_sen.append('.')
                else: remove_tag_sen.append(w)
            remove_tag_sen = [v for i, v in enumerate(remove_tag_sen) if i not in remove_indices]
            remove_tag_sen = ' '.join(remove_tag_sen)
            #remove_tag_words = [v for i, v in enumerate(words) if i not in remove_indices]
            #remove_tag_sen = ' '.join(remove_tag_words)
            sentences = sent_tokenize(remove_tag_sen)
            for s in sentences:
                if len(s.split())<=3:continue
                cur_doc.append([s.strip(),0])
                write_file.write(s.strip()+"\t"+str(0)+"\n")
                num_sen += 1
            '''

            sentences = sent_tokenize(line.strip())
            #sentences = [lines]
            for s in sentences:
                    num_sen += 1
                    sentence = s.split()
                    cur_sen_len = len(sentence)
                    if cur_sen_len > max_sen_len: max_sen_len = cur_sen_len
                    total_sen_len += cur_sen_len
                    #words = set(sentence)
                    for word in sentence:
                        vocab[word] += 1
                    #if  ("POS" in s or "/POS" in s): continue
                    if ("NEG" in s or "/NEG" in s):
                        s = s.strip()
                        s = s.replace('<NEG>','')
                        s = s.replace('</NEG>','')
                        s = s.replace('< NEG >','')
                        s = s.replace('< /NEG >','')
                        write_file.write(s+"\t"+str(1)+"\n")
                        cur_doc.append([s,1])
                        num_rational += 1
                        continue
                    if (cur_sen_len) <= 2: continue
                    cur_doc.append([" ".join(sentence).strip(),0])
                    write_file.write(" ".join(sentence).strip()+"\t"+str(0)+"\n")

            if len(cur_doc) > max_doc_len: max_doc_len = len(cur_doc)
            total_doc_len += len(cur_doc)
            datum  = {"y":0,
                              "text": cur_doc,
                              "split": int(neg_file.split('_')[1][0])}

            revs.append(datum)
            write_file.write("0")
            write_file.close()
    print "max sentence length: " + str(max_sen_len)
    print "max document length: " + str(max_doc_len)
    print "average sentence length: " + str(total_sen_len/num_sen)
    print "average document length: " + str(total_doc_len/num_doc)
    print "average number of rationals: " + str(num_rational/num_doc)
    return revs, vocab

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')            
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    w2v_file = sys.argv[1]     
    data_folder = ["movies/withRats_pos","movies/withRats_neg"]
    print "loading data...",        
    revs, vocab = build_data_cv(data_folder, clean_string=True)
    print "data loaded!"
    print "number of documents: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab,min_df=1)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab,min_df=1)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("movie_Doc.p", "wb"))
    print "dataset created!"