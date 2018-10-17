#%%
import keras
import numpy as np
import pandas as pd
from utils import *
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import train_test_split



#%%
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    ### START CODE HERE ###
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding(vocab_len, emb_dim, trainable=False)
    ### END CODE HERE ###

    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

#%%
def create_model(input_shape_target,input_shape_callout,word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    
    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    sentence_indices_target = Input(shape=input_shape_target, dtype=np.int32)

    sentence_indices_callout = Input(shape=input_shape_callout, dtype=np.int32)
    
    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings_target = embedding_layer(sentence_indices_target)   
    embeddings_callout = embedding_layer(sentence_indices_callout) 

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.

    shared_layer = LSTM(128, return_sequences=True)
    X_target = shared_layer(embeddings_target)
    X_callout = shared_layer(embeddings_callout)

    # Add dropout with a probability of 0.5
    X_target = Dropout(0.5)(X_target)
    X_callout = Dropout(0.5)(X_callout)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    shared_layer2 = LSTM(128)

    X_target = shared_layer2(X_target)
    X_callout = shared_layer2(X_callout)
    # Add dropout with a probability of 0.5
    X_target = Dropout(0.5)(X_target)
    X_callout = Dropout(0.5)(X_callout)

    X = keras.layers.concatenate([X_target,X_callout],axis = -1)




    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(1, activation='sigmoid')(X)
    
    # Create Model instance which converts sentence_indices into X.
    model = Model([sentence_indices_target,sentence_indices_callout], X)
    
    
    return model



#%%
def import_data(word_to_index):

    # import labeled data
    data_android = pd.read_csv(".\\data\\ReleaseData\\turker_annotated_task1\\android100.turker1.txt",sep = "	")
    data_android_matrix = np.asarray(data_android)
    data_ipad = pd.read_csv(".\\data\\ReleaseData\\turker_annotated_task1\\ipad100.turker1.txt",sep = "	")
    data_ipad_matrix = np.asarray(data_ipad)
    data_layoffs = pd.read_csv(".\\data\\ReleaseData\\turker_annotated_task1\\layoffs100.turker1.txt",sep = "	")
    data_layoffs_matrix = np.asarray(data_layoffs)
    data_twitter = pd.read_csv(".\\data\\ReleaseData\\turker_annotated_task1\\twitter.turker1.txt",sep = "	")
    data_twitter_matrix = np.asarray(data_twitter)   
    data = np.concatenate((data_android_matrix,data_ipad_matrix,data_layoffs_matrix,data_twitter_matrix),axis = 0)
    # Split target and callout
    X_target = data[:,2]
    X_callout = data[:,3]
    Y = data[:,4:]   

    # Prepocessing x
    lemmatizer = WordNetLemmatizer()

    for index,sentence in enumerate(X_target):
        tokenizer = RegexpTokenizer(r'\w+')
        word_list = tokenizer.tokenize(sentence)
        word_list_lemma = [lemmatizer.lemmatize(word.lower()) for word in word_list]
        word_list_lemma_indices = [word_to_index[word] if word in word_to_index.keys() else word_to_index["unk"] for word in word_list_lemma]
        X_target[index] = np.asarray(word_list_lemma_indices)
    
    for index,sentence in enumerate(X_callout):
        tokenizer = RegexpTokenizer(r'\w+')
        word_list = tokenizer.tokenize(sentence)
        word_list_lemma = [lemmatizer.lemmatize(word.lower()) for word in word_list]
        word_list_lemma_indices = [word_to_index[word] if word in word_to_index.keys() else word_to_index["unk"] for word in word_list_lemma]
        X_callout[index] = np.asarray(word_list_lemma_indices)
    
    X_target = pad_sequences(X_target)
    X_callout = pad_sequences(X_callout)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if Y[i][j] == "choice1":
                Y[i][j] = 1
            if Y[i][j] == "choice2":
                Y[i][j] = 0.5
            if Y[i][j] == "choice3":
                Y[i][j] = -0.5
            if Y[i][j] == "choice4":
                Y[i][j] = -1
            if Y[i][j] == "choice5":
                Y[i][j] = 0

    Y = np.nansum(Y,axis = 1,dtype = "float")
    for i in range(len(Y)):
        if Y[i] > 0.5:
            Y[i] = 1
        elif Y[i] < -0.5:
            Y[i] = 0
        else:
            Y[i] = -1
    return (X_target,X_callout,Y)    
    
    



#%%
# import Glove embeddings
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('.\\embedding\\glove.6B.50d.txt')

X_target,X_callout,Y = import_data(word_to_index)

X_target = X_target[np.where(Y != -1)]
X_callout = X_callout[np.where(Y != -1)]
Y = Y[np.where(Y != -1)]
print ("Data size:")
print (X_target.shape,X_callout.shape,Y.shape)
X_target_train,X_target_test,X_callout_train,X_callout_test,Y_train,Y_test =train_test_split(X_target,X_callout,Y,train_size = 0.8)


#%%
model = create_model((X_target.shape[1],),(X_callout.shape[1],), word_to_vec_map, word_to_index)
model.summary()

#%%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
model.fit([X_target_train,X_callout_train], Y_train, epochs = 30, batch_size = 32, shuffle=True)

#%%
Y_hat = model.predict([X_target_test,X_callout_test]).ravel()
Y_pred = [1 if prob >=0.5 else 0 for prob in Y_hat]

f1 = f1_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
print ('f1 = {}, precision = {}, recall = {}'.format(f1,precision,recall))

#%%
accuracy_score(Y_test, Y_pred)

