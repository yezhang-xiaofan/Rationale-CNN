# Rationale-CNN
This repository implements the model from paper: [Rationale-Augmented Convolutional Neural Networks for Text Classification](https://arxiv.org/pdf/1605.04469v2.pdf). 

Much of the code is modified from: https://github.com/yoonkim/CNN_sentence

## Preprocessing Data
You need to download [Pre-trained word2vec file](https://code.google.com/p/word2vec/). 
```
python process_data_doc.py path_to_word2vec
```
where path_to_word2ec is the location of the pre-trained word2vec file. 

## Train the rationale-CNN model
The training program is written to train on 8 folds, and test on the rest fold (there are totally 9 folds), so you need to specify which fold you want to test on. For example, if you want to test on 0th fold, you can run the following. 
```
python rationale_CNN.py -nonstatic -word2vec -0
```
Note that the optimal value of dropout rate on sentences when training document-level CNN might be different on different folds, and it is worth further tuning. 
