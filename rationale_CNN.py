"""
Much of the code is modified from https://github.com/yoonkim/CNN_sentence
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import warnings
import sys
import time
from conv_net_classes import *
import math
import util

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

def train_conv_net(datasets,
                   U,
                   idx_word_map,
                   img_w=300,
                   filter_hs=[3,4,5],
                   hidden_units=[100,2],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25,
                   batch_size=50,
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True,
                   sen_dropout_rate=[0.0],
                              whether_train_sen=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    rng = np.random.RandomState(3435)
    img_h = datasets[0][0][0].shape[0]-1
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch),('sentence dropout rate',sen_dropout_rate)]
    print parameters

    #define model architecture
    index = T.lscalar()
    x = T.tensor3('x')
    y = T.ivector('y')
    sen_x = T.matrix('sen_x')
    mark = T.matrix('mark')
    sen_y = T.ivector('sen_y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))],
                               allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0]*x.shape[1],1,x.shape[2],Words.shape[1]))
    sen_layer0_input = Words[T.cast(sen_x.flatten(),dtype='int32')].reshape((sen_x.shape[0],1,sen_x.shape[1],
                                                                             Words.shape[1]))
    conv_layers = []
    layer1_inputs = []
    Doc_length = datasets[0][0].shape[0]
    sen_layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,
                                        #image_shape=(batch_size*Doc_length, 1, img_h, img_w),
                                        image_shape=(None,1,img_h,img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)

        sen_layer1_input = conv_layer.predict(sen_layer0_input,None).flatten(2)
        sen_layer1_inputs.append(sen_layer1_input)

    layer1_input = T.concatenate(layer1_inputs,1)
    sen_layer1_input = T.concatenate(sen_layer1_inputs,1)

    hidden_units[0] = feature_maps*len(filter_hs)
    sen_hidden_units = [feature_maps*len(filter_hs),3]
    shaped_mark = T.flatten(mark)

    sen_classifier1 = MLPDropout(rng,input=sen_layer1_input,layer_sizes=sen_hidden_units,activations=activations,
                                 dropout_rates=sen_dropout_rate)
    sen_cost = sen_classifier1.dropout_negative_log_likelihood(sen_y)
    sen_pos_prob = T.max(sen_classifier1.predict_p(layer1_input)[:,np.array([0,2])],axis=1)
    prev_layer1_output,updates = theano.scan(fn=lambda i,x:x[i*Doc_length:i*Doc_length+Doc_length],
                                             sequences=[T.arange(batch_size)],
                                             non_sequences=layer1_input*(sen_pos_prob.dimshuffle(0,'x'))*(shaped_mark.dimshuffle(0,'x'))
                                            )
    layer1_output = T.sum(prev_layer1_output,axis=1)
    classifier = MLPDropout(rng, input=layer1_output, layer_sizes=hidden_units, activations=activations,
                            dropout_rates=dropout_rate)

    #define parameters of the model and update functions using adadelta
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]

    #add sentence level parameters
    sen_params = sen_classifier1.params
    for conv_layer in conv_layers:
        sen_params += conv_layer.params
    if non_static:
        sen_params += [Words]

    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
    sen_grad_updates = sgd_updates_adadelta(sen_params,sen_cost,lr_decay,1e-6,sqr_norm_lim)
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
    #extra data (at random)
    np.random.seed(3435)
    train_mask = np.zeros((datasets[0].shape[0],datasets[0].shape[1]),dtype='float32')  ##doc length * number of documnts
    test_mask = np.zeros((datasets[2].shape[0],datasets[2].shape[1]),dtype='float32')

    #set the mask
    for i in range(datasets[0].shape[0]):
        for j in range(datasets[0][i].shape[0]):
            if np.count_nonzero(datasets[0][i][j]) != 0:
                train_mask[i][j] = 1.0

    for i in range(datasets[2].shape[0]):
        for j in range(datasets[2][i].shape[0]):
            if np.count_nonzero(datasets[2][i][j])!=0:
                test_mask[i][j] = 1.0

    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        permuted_index = np.random.permutation(range(datasets[0].shape[0]))
        permuted_index = np.append(permuted_index,permuted_index[:extra_data_num])
        new_data=datasets[0][permuted_index]
    else:
        permuted_index = np.random.permutation(range(datasets[0].shape[0]))
        new_data = datasets[0][permuted_index]

    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets
    train_set_y = datasets[1][permuted_index]
    test_set_x,test_set_y = shared_dataset((datasets[2][:,:,:-1],datasets[3]))
    test_set_mark = theano.shared(test_mask.astype(theano.config.floatX))

    train_mask = train_mask[permuted_index]
    train_set_mark = train_mask[:n_train_batches*batch_size]
    train_set_mark = theano.shared(train_set_mark.astype(theano.config.floatX))

    train_set_with_sen_label = new_data[:n_train_batches*batch_size]
    val_set_with_sen_label = new_data[n_train_batches*batch_size:]

    train_set = new_data[:n_train_batches*batch_size,:,:-1]
    train_set_label = train_set_y[:n_train_batches*batch_size]

    val_set = new_data[n_train_batches*batch_size:,:,:-1]
    val_set_label = train_set_y[n_train_batches*batch_size:]
    val_set_mark = train_mask[n_train_batches*batch_size:]
    val_set_mark = theano.shared(val_set_mark.astype(theano.config.floatX))

    train_set_x, train_set_y = shared_dataset((train_set,train_set_label))

    val_set_x, val_set_y = shared_dataset((val_set,val_set_label))

    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size],
         mark:val_set_mark[index*batch_size:(index+1)*batch_size]},
                                allow_input_downcast=True)

    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size],
                                 mark:train_set_mark[index*batch_size:(index+1)*batch_size]},
                                 allow_input_downcast=True)

    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size],
          mark:train_set_mark[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)

    test_pred_layers = []
    test_size = datasets[2].shape[0]
    test_batch_size = 1
    n_test_batches = int(math.ceil(test_size/float(test_batch_size)))
    print "number of test batches: " + str(n_test_batches)
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0]*x.shape[1],1,
                                                                          x.shape[2],Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_batch_size*Doc_length)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_sen_prob = T.max(sen_classifier1.predict_p(test_layer1_input)[:,np.array([0,2])],axis=1)
    test_sen_prob_to_sen, updates = theano.scan(fn = lambda i ,x : x[i*Doc_length:i*Doc_length+Doc_length],
                                       sequences=[T.arange(test_batch_size)],
                                       non_sequences=test_sen_prob)

    sorted_index = T.argsort(test_sen_prob_to_sen*shaped_mark,axis=-1)[:,-5:]
    sorted_sentence,updates = theano.scan(fn=lambda i, y: y[i,sorted_index[i],:],
                sequences=[T.arange(sorted_index.shape[0])],
                non_sequences=x
                )
    sorted_prob,updates = theano.scan(fn=lambda  i, z: z[i,sorted_index[i]],
                    sequences = [T.arange(sorted_index.shape[0])],
                     non_sequences= test_sen_prob_to_sen
                    )

    sorted_sentence_value = theano.function([index],sorted_sentence,allow_input_downcast=True,
                                            givens={x:test_set_x[index*test_batch_size:(index+1)*test_batch_size],
                                                 mark:test_set_mark[index*test_batch_size:(index+1)*test_batch_size]})


    sorted_prob_val = theano.function([index],sorted_prob,allow_input_downcast=True,
                                      givens={
                                          x:test_set_x[index*test_batch_size:(index+1)*test_batch_size],
                                                    mark:test_set_mark[index*test_batch_size:(index+1)*test_batch_size]
                                      })

    test_layer1_output, updates = theano.scan(fn=lambda i, x :x[i*Doc_length:i*Doc_length+Doc_length],
                                          sequences=[T.arange(test_batch_size)],
                non_sequences=test_layer1_input*(test_sen_prob.dimshuffle(0,'x'))*(shaped_mark.dimshuffle(0,'x'))
                                              )
    test_layer1_output = T.sum(test_layer1_output,axis=1)
    test_y_pred = classifier.predict(test_layer1_output)
    test_error = T.mean(T.neq(test_y_pred, y))

    test_model_all = theano.function([index], test_error, allow_input_downcast = True,
                                     givens={
                                         x:test_set_x[index*test_batch_size:(index+1)*test_batch_size],
                                         y:test_set_y[index*test_batch_size:(index+1)*test_batch_size],
                                     mark:test_set_mark[index*test_batch_size:(index+1)*test_batch_size],})

    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0
    cost_epoch = 0
    sen_batch_size = 50
    best_sen_param = []
    for p in sen_params:
        best_sen_param.append(theano.shared(p.get_value()))

    best_sen_val = 0.0
    if whether_train_sen == True:
        print 'pre-train on sentences'
        while(epoch < 20):
            sen_costs = []
            train_sen = train_set_with_sen_label
            train_sentences = util.doc_to_sen(train_sen)
            train_sentences = util.remove(train_sentences)
            train_sentences = util.downsample_three(train_sentences)
            print "positive sentences after sampling: " + str(np.sum(train_sentences[:,-1]==2))
            print "negative sentences after sampling: " + str(np.sum(train_sentences[:,-1]==0))
            print "neutral sentences after sampling: " + str(np.sum(train_sentences[:,-1]==1))
            train_sentences = np.random.permutation(train_sentences)
            if train_sentences.shape[0]%sen_batch_size!=0:
                        extra_data_num = sen_batch_size - train_sentences.shape[0] % sen_batch_size
                        extra_index = np.random.permutation(range(train_sentences.shape[0]))[:extra_data_num]
                        train_sentences = np.vstack((train_sentences,train_sentences[extra_index]))
            train_sen_x, train_sen_y = shared_dataset((train_sentences[:,:-1],train_sentences[:,-1]))
            train_sen_model = theano.function([index],sen_cost,updates=sen_grad_updates,
                                             givens={
                                sen_x:train_sen_x[index*sen_batch_size:(index+1)*sen_batch_size],
                                 sen_y: train_sen_y[index*sen_batch_size:(index+1)*sen_batch_size]})

            n_train_sen_batches = train_sentences.shape[0]/sen_batch_size
            for minibatch_index_1 in np.random.permutation(range(n_train_sen_batches)):
                        cur_sen_cost = train_sen_model(minibatch_index_1)
                        sen_costs.append(cur_sen_cost)
                        set_zero(zero_vec)

            print "training sentence cost: " + str(sum(sen_costs)/len(sen_costs))
            val_sen = val_set_with_sen_label
            val_sentences = util.doc_to_sen(val_sen)
            val_sentences = util.remove(val_sentences)
            print "positive sentences in the validation set: " + str(np.sum(val_sentences[:,-1]==2))
            print "negative sentences in the validation set: " + str(np.sum(val_sentences[:,-1]==0))
            print "neutral sentences in the validation set: " + str(np.sum(val_sentences[:,-1]==1))
            val_sen_x,val_sen_y = shared_dataset((val_sentences[:,:-1],val_sentences[:,-1]))
            val_sen_model = theano.function([],sen_classifier1.errors(sen_y),
                                                givens={
                                                    sen_x:val_sen_x,sen_y:val_sen_y})
            val_accuracy = 1 - val_sen_model()
            print "validation sentence accuracy: " + str(val_accuracy)
            if val_accuracy > best_sen_val:
                 best_sen_val = val_accuracy
                 for i,p in enumerate(best_sen_param):
                     p.set_value(sen_params[i].get_value())
            epoch = epoch + 1
        for i,sp in enumerate(sen_params):
            sp.set_value(best_sen_param[i].get_value())

    epoch = 0
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch,
                                                        time.time()-start_time, train_perf * 100., val_perf*100.))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = [test_model_all(i) for i in xrange(n_test_batches)]
            test_perf = 1- np.sum(test_loss)/float(test_size)
            print "best test performance so far: " + str(test_perf)
            print test_loss
    test_loss = [test_model_all(i) for i in xrange(n_test_batches)]
    new_test_loss = []
    for i in test_loss:
        new_test_loss.append(np.asscalar(i))
    test_loss = new_test_loss
    correct_index = np.where(np.array(test_loss)==0)[0]
    count_pos = 0
    test_labels = np.array(datasets[3])
    print "negative estimated rationales: "
    print len(idx_word_map)
    for c in correct_index:
        if test_labels[c] == 1:continue
        print util.convert(sorted_sentence_value(c)[0],idx_word_map)
        print sorted_prob_val(c)
        count_pos += 1
        if count_pos == 2:
            break

    count_neg = 0
    print "positive estimated rationales: "
    for c in correct_index:
        if test_labels[c] == 0:continue
        print util.convert(sorted_sentence_value(c)[0],idx_word_map)
        print sorted_prob_val(c)
        count_neg += 1
        if count_neg == 2:
            break
    return test_perf

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to

def get_idx_from_sent(sent, word_idx_map, max_sen_len=51, max_Doc_len=50, filter_h=5,doc_label=1):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    doc = np.zeros((max_Doc_len,max_sen_len+2*(filter_h-1)+1))
    doc_len = len(sent)
    sen_count = 0
    for i in range(doc_len):
        cur_sen = sent[i][0]
        cur_sen_label = sent[i][1]
        if cur_sen_label == 0: continue
        x = []
        pad = filter_h - 1
        for i in xrange(pad):
            x.append(0)
        words = cur_sen.split()
        for word in words:
            if len(x) >= max_sen_len + 2*pad: break
            if word in word_idx_map:
                x.append(word_idx_map[word])
        while len(x) < max_sen_len+2*pad:
            x.append(0)
        doc[sen_count,:-1] = np.array(x)
        doc[sen_count,-1] = 0 if doc_label == 0 else 2  ######negative or positive
        sen_count += 1

    for i in range(doc_len):
        if sen_count >= max_Doc_len: break
        cur_sen = sent[i][0]
        cur_sen_label = sent[i][1]
        if cur_sen_label == 1: continue   ##neutral sentences
        x = []
        pad = filter_h - 1
        for i in xrange(pad):
            x.append(0)
        words = cur_sen.split()
        for word in words:
            if len(x) >= max_sen_len + 2*pad: break
            if word in word_idx_map:
                x.append(word_idx_map[word])
        while len(x) < max_sen_len+2*pad:
            x.append(0)
        doc[sen_count,:-1] = np.array(x)
        doc[sen_count,-1] = 1
        sen_count += 1

    return doc

def get_idx_from_doc(sent, word_idx_map, max_l=51,k=300,filter_h=5):
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    for s in sent:
        sentence = s[0]
        words = sentence.split()
        for word in words:
            if len(x) >= max_l + 2*pad: break
            if word in word_idx_map:
                x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_sen_len=40,max_Doc_len=40, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    train_y,test_y = [],[]
    for rev in revs:
        if rev['split'] == 9: continue     ###########note that we don't use fold 9!!!!!!!!!!!
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_sen_len, max_Doc_len,filter_h,rev["y"])
        print rev["y"]
        if rev["split"]==cv:
            test.append(sent)
            test_y.append(rev["y"])
        else:
            train.append(sent)
            train_y.append(rev["y"])

    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    print "train shape: " + str(train.shape)
    print "test shape: " + str(test.shape)
    print "positive documents in test set: " + str(sum(test_y))
    print "positive documents in training set: " + str(sum(train_y))
    return [train,np.array(train_y),test, np.array(test_y)]


if __name__=="__main__":
    print "loading data...",
    x = cPickle.load(open("movie_Doc.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    idx_word_map = {}
    for w in word_idx_map:
        idx_word_map[word_idx_map[w]] = w
    print "data loaded!"
    mode= sys.argv[1]
    word_vectors = sys.argv[2]
    test_fold = int(sys.argv[3])
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []
    r = range(0,10)
    for i in r:
        if i == 9 or i!= test_fold:continue
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_sen_len=30,max_Doc_len=40, filter_h=5)
        perf = train_conv_net(datasets,
                              U,
                              idx_word_map,
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              conv_non_linear="relu",
                              hidden_units=[20,2],
                              shuffle_batch=True,
                              n_epochs=25,
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=50,
                              dropout_rate=[0.5],
                              sen_dropout_rate=[0.5],
                              whether_train_sen=True)
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)
    print str(np.mean(results))
