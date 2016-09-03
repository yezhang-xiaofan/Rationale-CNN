__author__ = 'zhangye'
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TreebankWordTokenizer

def doc_to_sen(X_train):
    '''
    :X_train: shape: (number of docs * number of sentences * (sentence length + 1))
    :return:
    '''
    num_Doc = X_train.shape[0]
    Doc_len = X_train.shape[1]
    sentences = X_train.reshape((num_Doc*Doc_len,X_train.shape[2]))
    return sentences

def downsample(X_train):
    '''

    :param X_train:
    :param X_label:
    :return: equal sized positive and negative sentences
    '''
    num_pos = np.count_nonzero(X_train[:,-1])
    num_neg = X_train.shape[0]-num_pos
    #print train[labels==1]
    print "number of positive sentences before downsampling: "
    print num_pos
    print "number of negative sentences before downsampling: "
    print num_neg
    if num_neg>num_pos:
        neg_index =  np.nonzero((X_train[:,-1]==0))
        pos_index = np.nonzero((X_train[:,-1] == 1))
        #print pos_index
        #print neg_index
        select = X_train[np.random.permutation(neg_index[0])]  ###select negative sentences 
        new_select = np.zeros((num_pos,X_train.shape[1]))
        count_neg = 0
        for i in range(num_neg):
            if np.sum(select[i])==0: continue
            new_select[count_neg] = select[i]
            count_neg += 1
            if count_neg == num_pos:
                break
        #print select
        train = np.vstack((X_train[pos_index[0]],new_select))
        return train

def downsample_three(X_train):
    '''
    :param X_train:
    :param X_label:
    :return: equal sized positive and negative sentences
    '''
    num_neg = np.count_nonzero(X_train[:,-1]==0)
    num_neu = np.count_nonzero(X_train[:,-1]==1)
    num_pos = np.count_nonzero(X_train[:,-1]==2)
    #print train[labels==1]
    print "number of positive sentences before sampling: "
    print num_pos
    print "nubmer of negative sentences before sampling: "
    print num_neg
    print "number of neutral sentences before sampling: "
    print num_neu

    if num_neg>num_pos:
        '''
        track = 0
        while(train[track%train.shape[0],-1]==0): track+=1
        new_train = np.zeros((1,train.shape[1]))
        while True:
            sample = np.random.binomial(1,0.5,1)
            if sample:
                new_train = np.vstack((new_train,train[track%train.shape[0],:]))
            while(train[track%train.shape[0],-1]==0): track+=1
            if new_train.shape[0]-1+num_pos == num_neg: break
        #print new_train
        train = np.vstack((train,new_train[1:,]))
        '''
        neg_index =  np.nonzero((X_train[:,-1]==0))
        pos_index = np.nonzero(X_train[:,-1]==2)
        neu_index = np.nonzero(X_train[:,-1]==1)
        select = X_train[np.random.permutation(neg_index[0])]    ###random shuffle negative sentences
        new_select = np.zeros((num_pos,X_train.shape[1]))
        count_neg = 0
        for i in range(num_neg):
            if np.sum(select[i])==0: continue
            new_select[count_neg] = select[i]
            count_neg += 1
            if count_neg == num_pos:
                break

        select_neu = X_train[np.random.permutation(neu_index[0])]
        new_neu_select = np.zeros((num_pos,X_train.shape[1]))
        count_neu = 0
        for i in range(num_neu):
            if np.sum(select_neu[i])==0: continue
            new_neu_select[count_neu] = select_neu[i]
            count_neu += 1
            if count_neu == num_pos:
                break
        #print select
        train = np.vstack((new_select,new_neu_select,X_train[pos_index]))

    else:   ##pos > neg
        neg_index =  np.nonzero((X_train[:,-1]==0))
        pos_index = np.nonzero(X_train[:,-1]==2)
        neu_index = np.nonzero(X_train[:,-1]==1)
        select = X_train[np.random.permutation(pos_index[0])]
        new_select = np.zeros((num_neg,X_train.shape[1]))
        count_pos = 0
        for i in range(num_pos):
            if np.sum(select[i])==0: continue
            new_select[count_pos] = select[i]
            count_pos += 1
            if count_pos == num_neg:
                break
        select_neu = X_train[np.random.permutation(neu_index[0])]

        new_neu_select = np.zeros((num_neg,X_train.shape[1]))
        count_neu = 0
        for i in range(num_neu):
            if np.sum(select_neu[i])==0: continue
            new_neu_select[count_neu] = select_neu[i]
            count_neu += 1
            if count_neu == num_neg:
                break
        #print select
        train = np.vstack((new_select,new_neu_select,X_train[neg_index]))
    return train
def create_batch(sentences,batch_size):
    '''

    :param sentences:
    :return: train batches, val batches
    '''
    if sentences.shape[0] % batch_size > 0:
        extra_data_num = batch_size - sentences.shape[0] % batch_size
        train_set = np.random.permutation(sentences)
        extra_data = train_set[:extra_data_num]
        new_data=np.append(sentences,extra_data,axis=0)
    else:
        new_data = sentences
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.floor(n_batches*0.9))
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]
    return train_set,val_set

def generate_voc(X,min_df=4):
    '''
    :param X: list of documents
    :return:
    '''
    word_Count = {}
    for x in X:
        x = x.split()
        for word in x:
            if word in word_Count:
                word_Count[word] += 1
            else:
                word_Count[word] = 1
    vocab = {}
    for word in word_Count:
        if word_Count[word] >= min_df:
            vocab[word] = word_Count[word]
    return vocab

def remove(sentences):  ###remove "false" sentences
    new_indices = []
    for i in range(sentences.shape[0]):
        if np.sum(sentences[i])!=0:   ##real sentences !!!
            new_indices.append(i)
    return sentences[np.array(new_indices)]

def convert(sentences,idx_to_word):
    '''
    :param sentences:
    :return:
    '''
    final_res = []
    for i in range(sentences.shape[0]):
        if np.sum(sentences[i])==0: continue
        words = []
        for j in range(sentences.shape[1]-1):
            if sentences[i][j] == 0: continue
            words.append(idx_to_word[sentences[i][j]])
        words.append(sentences[i][-1])
        final_res.append(words)
    return final_res
