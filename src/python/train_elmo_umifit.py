from collections import Counter
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import argparse
import torch
# from fastai import *
# from fastai.text import * 

def import_data(path):

    # import labeled data
    data_android = pd.read_csv(os.path.join(path,"android100.turker1.txt"),sep = "\t")
    data_android_matrix = np.asarray(data_android)
    data_ipad = pd.read_csv(os.path.join(path,"ipad100.turker1.txt"),sep = "\t")
    data_ipad_matrix = np.asarray(data_ipad)
    data_layoffs = pd.read_csv(os.path.join(path,"layoffs100.turker1.txt"),sep = "\t")
    data_layoffs_matrix = np.asarray(data_layoffs)
    data_twitter = pd.read_csv(os.path.join(path,"twitter.turker1.txt"),sep = "\t")
    data_twitter_matrix = np.asarray(data_twitter)   
    data = np.concatenate((data_android_matrix,data_ipad_matrix,data_layoffs_matrix,data_twitter_matrix),axis = 0)

    # Split target and callout
    X_target = data[:,2]
    X_callout = data[:,3]
    Y = data[:,4:]   

    # Prepocessing x
    
   
    target = np.asarray(X_target)
    callout = np.asarray(X_callout)
    label = []
    for i in range(Y.shape[0]):
        count = Counter(Y[i])
        choice,_ = count.most_common()[0]
        if choice in ["choice1","choice2"]:
            label.append(1)
        elif choice in ["choice3","choice4"]:
            label.append(0)
        else:
            label.append(-1)  
            

    
    df = {"target_sentence":target,"callout_sentence":callout,"label":label}
    df = pd.DataFrame.from_dict(df)[["callout_sentence","target_sentence","label"]]
    df = df[df["label"] != -1]   
   
    return df

def get_predictions(estimator, input_fn):
	return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]


def main(args):

    print (args)


    df = import_data(args.data_path)
    df = df.sample(frac=1).reset_index(drop=True)
    train_df,test_df = train_test_split(df,train_size = 0.7,shuffle = True)

    if args.mode == "umfit":


        df.to_csv("./lm_data/test.csv",index = None)

        # Language model data
        data_lm = TextLMDataBunch.from_csv("./lm_data","test.csv")
        # Classifier model data
        data_class = TextClasDataBunch.from_csv("./lm_data","test.csv",valid_pct = 0.3, label_cols = 2,vocab=data_lm.train_ds.vocab, bs=32)


        learn = language_model_learner(data_lm, pretrained_model=URLs.WT103, drop_mult=0.2,callback_fns=ShowGraph)
        learn.unfreeze()
        learn.fit_one_cycle(140, 1e-3)

        learn.save_encoder('ft_enc')
        learn = text_classifier_learner(data_class, drop_mult=0.3,callback_fns=ShowGraph)
        learn.load_encoder('ft_enc')

        learn.unfreeze()
        learn.fit_one_cycle(100, 1e-4)

        prediction = learn.get_preds(ds_type = "Valid")
        y_pred = prediction[0].numpy()
        y_pred = np.argmax(y_pred,axis = 1)
        y_true = prediction[1].numpy()

        target_names = ['Disagree','Agree']
        print (classification_report(y_true, y_pred,target_names=target_names))

    else:        

        # Training input on the whole training set with no limit on training epochs.
        train_input_fn = tf.estimator.inputs.pandas_input_fn(
            train_df, train_df["label"], batch_size = 32,num_epochs=None, shuffle=True)

        # Prediction on the whole training set.
        predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
            train_df, train_df["label"], batch_size = 32,shuffle=False)
        # Prediction on the test set.
        predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
            test_df, test_df["label"], batch_size = 32,shuffle=False)


        tf.logging.set_verbosity(tf.logging.INFO)


        if args.mode == "elmo":
            embedded_callout = hub.text_embedding_column(
                key="callout_sentence", 
                module_spec="https://tfhub.dev/google/elmo/2",
                trainable = True)


            embedded_target = hub.text_embedding_column(
                key="target_sentence", 
                module_spec="https://tfhub.dev/google/elmo/2",
                trainable = True)
        else:
            embedded_callout = hub.text_embedding_column(
                key="callout_sentence", 
                module_spec="https://tfhub.dev/google/universal-sentence-encoder-large/3")


            embedded_target = hub.text_embedding_column(
                key="target_sentence", 
                module_spec="https://tfhub.dev/google/universal-sentence-encoder-large/3")

        #cosine decay with restart


        #cos decay
        estimator = tf.estimator.DNNClassifier(
            feature_columns=[embedded_callout,embedded_target],
            hidden_units=[512, 256],
            activation_fn = tf.nn.leaky_relu,
            dropout = 0.6,
            optimizer=lambda: tf.train.AdamOptimizer(
                learning_rate=tf.train.cosine_decay(
                    learning_rate=0.01,
                    global_step=tf.train.get_global_step(),
                    decay_steps=100)),batch_norm = True)



        estimator.train(input_fn=train_input_fn, steps = 5000)

        train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
        test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

        print("Training set accuracy: {accuracy}".format(**train_eval_result))
        print("Test set accuracy: {accuracy}".format(**test_eval_result))
        Y_pred = get_predictions(estimator, predict_test_input_fn)
        target_names = ['Disagree', 'Agree']
        print (classification_report(test_df["label"], Y_pred,target_names=target_names))
        test_df["predict"] = Y_pred
        test_df[test_df["label"] != test_df["predict"]].to_csv("error.csv")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LRCT')

    parser.add_argument('--data_path', type=str, action='store', help='the path where store training data', default="./data")   

    parser.add_argument('--mode', type=str,action='store', help='elmo,universal,umfit', default="elmo")

    args = parser.parse_args()

    main(args)