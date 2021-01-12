

import pandas as pd
import numpy as np
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
#from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  classification_report

data = pd.read_csv('C:\\Users\\Ashish\\Desktop\\mails proj 3\\emails.csv')

data.isna().sum()
#data.drop(data['Unnamed: 0'],inplace= True,axis=0)
#data.dr
data.drop('Unnamed: 0',inplace= True,axis=1)
data.info()
data.describe()
data['Class'].value_counts()
data['Message-ID'].dtype

data.drop('filename',inplace= True,axis=1)
data.drop('Message-ID',inplace= True,axis=1)

sns.countplot(data['Class'])


A_class = data[data.Class=='Abusive']
B_class = data[data.Class=='Non Abusive']

def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
data['content']=data['content'].apply(lambda x: remove_URL(x))

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text
data['content']=data['content'].apply(lambda x: remove_punct(x))

def remove_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])         
    return text
data['content']=data['content'].apply(lambda x: remove_numbers(x))
 
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)
data['content']=data['content'].apply(lambda x: remove_html(x))



from wordcloud import WordCloud, STOPWORDS 

stop_words = []
with open("C:/Users/Ashish/Desktop/datasci_assignment/text mining\\stop.txt") as f:
    stop_words = f.read()
    

def cleaning_text(i):
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

# resampling
    
A_class = resample(A_class,replace=True,n_samples=44666,random_state=123)
data = pd.concat([A_class,B_class])
data.Class.value_counts()

lbec = LabelEncoder()
data["Class"] = lbec.fit_transform(data["Class"])












from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer




# Creating Document Term Matrix
# creating a matrix of token counts for the entire text document 

def split_into_words(i):
    return [word for word in i.split(" ")]




# Preparing email texts into word count matrix format 
data_bow = CountVectorizer(analyzer=split_into_words).fit(data.content)

# ["mailing","body","texting"]
# ["mailing","awesome","good"]

# ["mailing","body","texting","good","awesome"]



#        "mailing" "body" "texting" "good" "awesome"
#  0          1        1       1        0       0
 
#  1          1        0        0       1       1    

# For all messages
all_data_matrix = data_bow.transform(data.content)






y = data.Class
X =all_data_matrix
 
# Train model
clf_4 = RandomForestClassifier()
clf_4.fit(X, y)
 
# Predict on training set

from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


pred_y_4 = clf_4.predict(X)
 

print( np.unique( pred_y_4 ) )

# How's our accuracy?
from sklearn.metrics import accuracy_score
print( accuracy_score(y, pred_y_4) )


from sklearn.metrics import precision_score,recall_score, confusion_matrix, classification_report,accuracy_score, f1_score

print ('\n clasification report:\n', classification_report(y,pred_y_4))



import joblib
import pickle


filename= 'mails_model111.pickle'
pickle.dump(clf_4,open('mails_model111.pkl', 'wb'))
