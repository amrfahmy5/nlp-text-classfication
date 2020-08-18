import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec
import string
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split 
from textblob import Word
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import preprocessing 

dataset = pd.read_csv("Tweets.csv")


y = dataset["airline_sentiment"]#postive natural negative
x = dataset["text"]#tweet psot
def preprocess(x_col):
    #displaying the list of stopwords
    stop = stopwords.words('english')   
    x_col = x_col.apply(lambda x:' '.join(x.lower() for x in x.split()))
    x_col= x_col.apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))
    x_col= x_col.str.replace('[^\w\s]','')
#    x_col= x_col.str.replace('@\w+','') 
    x_col= x_col.apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
    x_col = x_col.apply(lambda x:' '.join(x for x in x.split() if not x in stop))
    x_col = x_col.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return x_col
x = preprocess(x)
label_encoder = preprocessing.LabelEncoder() 
y_labeled= label_encoder.fit_transform(y)#convert y to 0 1 2
#  
parser = re.compile("@?\\w+")
#x_splitted  = [ parser.findall(i)  for i in  x ]
x_splitted  = [ i.split()  for i in  x ]#make the sentece into word
n = 500
def get_average(ll):
    npArray = np.zeros(n)
    for j in ll :
        try :
             npArray +=model_ted.wv.__getitem__(j)
        except Exception as e:
            print("[BAD WORD] :" , e)
    return npArray / len(ll)

model_ted =Word2Vec(sentences=x_splitted,size=n,min_count=1,workers=10,sg=1,hs=0)
c1 =0 
c2 = 0
x_average = []
#make average for each sentence(cont np array of averge for each word)
for i in x_splitted :
    x_average.append( get_average(i))


#splite the 3 type of y to be more honst (to make a 80% for train from each type)
x_positive = []
y_positive =[]
x_negative = []
y_negative = []
x_normal = []
y_normal = []

for i in range(len(y_labeled)):
    if y_labeled[i] == 2:
        x_positive.append(x_average[i])
        y_positive.append(y_labeled[i])
    elif y_labeled[i] == 0:
        x_negative.append(x_average[i])
        y_negative.append(y_labeled[i])
    elif y_labeled[i] == 1:
        x_normal.append(x_average[i])
        y_normal.append(y_labeled[i])
        

    
X_train_P, X_test_P, y_train_P, y_test_P = train_test_split(x_positive, y_positive, test_size=0.20)
X_train_N, X_test_N, y_train_N, y_test_N = train_test_split(x_negative, y_negative, test_size=0.20)
X_train_R, X_test_R, y_train_R, y_test_R = train_test_split(x_normal, y_normal, test_size=0.20)

X_train = X_train_N + X_train_R + X_train_P
y_train = y_train_N + y_train_R + y_train_P

X_test = X_test_N + X_test_R + X_test_P
y_test = y_test_N + y_test_R + y_test_P

#build model
from sklearn.linear_model   import LogisticRegression
classifier = LogisticRegression(random_state=1).fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("accuracy:",metrics.accuracy_score(y_test, y_pred)*100 , "%")
labels = {
    0 : "Negative",
    1 : "Normal",
    2: "Positive"
        }
txt = input("Tweet Now ...")
tokens  = parser.findall(txt)#splite each word to be a term in array
# example tweet man ----> [example,tweet,man]
avg = get_average(tokens)#get average for each word
#classifier.predict ----> return an array of 0 1 2 as detected in labels
print(labels[classifier.predict([avg])[0]])

