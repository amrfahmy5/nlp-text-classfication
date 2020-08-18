import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec
import string
from nltk.corpus import stopwords
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split 
from textblob import Word
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import preprocessing 


class Embedding:
    def __init__(self):
        self.STOPWORDS = set(stopwords.words('english'))
        self.dataset = pd.read_csv("Tweets.csv")
        self.y = self.dataset["airline_sentiment"]
        self.x = self.dataset["text"]
        self.preprocess(self.x)
        self.n = 200
        self.x_splitted  = [ i.split()  for i in  self.x ]
        self.label_encoder = preprocessing.LabelEncoder() 
        self.y_labeled= self.label_encoder.fit_transform(self.y)     
        self.model_ted = None
        self.x_average = []
        self.z  = 0
    def preprocess(self,x_col):
        """
        preprocess the tweets and clean it
        
        """
        stop = stopwords.words('english')   
        x_col = x_col.apply(lambda x:' '.join(x.lower() for x in x.split()))
        x_col= x_col.apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))
        x_col= x_col.str.replace('[^\w\s]','')
        x_col= x_col.apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
        x_col = x_col.apply(lambda x:' '.join(x for x in x.split() if not x in stop))
        x_col = x_col.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        return x_col
    
    
    def get_average(self , ll):
        npArray = np.zeros(self.n)
        for j in ll :
            try :
                 npArray +=self.model_ted.wv.__getitem__(j)
            except Exception as e:
                print("[BAD WORD] :" , e)
        return npArray / len(ll)
    
    
    def build_model(self):
        self.model_ted = Word2Vec(sentences=self.x_splitted,size=self.n,min_count=1,workers=10,sg=1,hs=0)
        for i in self.x_splitted :
            self.x_average.append(self.get_average(i))
    
    
    def spliting_data(self):
        x_positive = []
        y_positive =[]
        x_negative = []
        y_negative = []
        x_normal = []
        y_normal = []
        
        for i in range(len(self.y_labeled)):  
            if self.y_labeled[i] == 2:
                x_positive.append(self.x_average[i])
                y_positive.append(self.y_labeled[i])
            elif self.y_labeled[i] == 0:
                x_negative.append(self.x_average[i])
                y_negative.append(self.y_labeled[i])
            elif self.y_labeled[i] == 1:
                x_normal.append(self.x_average[i])
                y_normal.append(self.y_labeled[i])
                
                    
        X_train_P, X_test_P, y_train_P, y_test_P = train_test_split(x_positive, y_positive, test_size=0.20)
        X_train_N, X_test_N, y_train_N, y_test_N = train_test_split(x_negative, y_negative, test_size=0.20)
        X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(x_normal, y_normal, test_size=0.20)
        
        X_train = np.array(X_train_N + X_train_R + X_train_P)
        y_train = np.array(y_train_N + y_train_R + y_train_P)

        X_test = np.array(X_test_N + X_test_R + X_test_P)
        y_test = np.array(y_test_N + y_test_R + y_test_P)
        shuffle(X_train,y_train)
        shuffle(X_test,y_test)
         
        return X_train, X_test , y_train , y_test 
        