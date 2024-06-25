#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[16]:


df=pd.read_csv("spam (1).csv",encoding='latin-1')
df


# In[17]:


df.head()


# In[18]:


df.tail()


# In[19]:


df.info()


# In[20]:


df.describe


# In[22]:


df.shape


# In[26]:


columns=df.columns


# In[28]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[29]:


df.sample(5)


# In[30]:


df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[32]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[34]:


df['target']=encoder.fit_transform(df['target'])


# In[35]:


df.head()


# In[36]:


df.isnull().sum()


# In[37]:


df.duplicated().sum()


# In[40]:


df=df.drop_duplicates(keep='first')


# In[41]:


df.duplicated().sum()


# In[42]:


df.shape


# In[45]:


df['target'].value_counts()


# In[46]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[49]:


import nltk
get_ipython().system('pip install nltk')


# In[50]:


nltk.download('punkt')


# In[51]:


df['num_characters']=df['text'].apply(len)


# In[52]:


df.head()


# In[55]:


df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[56]:


df.head()


# In[58]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[59]:


df.head()


# In[60]:


df[['num_characters','num_words','num_sentences']].describe()


# In[61]:


df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[62]:


import seaborn as sns


# In[63]:


df[df['target'] ==0]['num_characters']


# In[64]:


sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'])


# In[65]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'])


# In[66]:


sns.pairplot(df,hue='target')


# In[ ]:





# In[92]:


def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return  " ".join(y)


# In[95]:


transform_text('I loved the yt lectures on ml presentation')


# In[87]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[84]:


import string
string.punctuation


# In[98]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('caring')


# In[100]:


df['transformed_text']=df['text'].apply(transform_text)


# In[101]:


df['transformed_text']


# In[102]:


df.head()


# In[112]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[113]:


spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[115]:


plt.figure(figsize=(15,8))
plt.imshow(spam_wc)


# In[116]:


ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[117]:


plt.figure(figsize=(15,8))
plt.imshow(ham_wc)


# In[118]:


df.head()


# In[122]:


spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[123]:


len(spam_corpus)


# In[129]:


from collections import Counter
common_words = Counter(spam_corpus).most_common(30)
df_common_words = pd.DataFrame(common_words, columns=['word', 'count'])
sns.barplot(x='word', y='count', data=df_common_words)
plt.xticks(rotation='vertical')
plt.show()


# In[130]:


ham_corpus = []
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[131]:


len(ham_corpus)


# In[132]:


from collections import Counter
common_words = Counter(ham_corpus).most_common(30)
df_common_words = pd.DataFrame(common_words, columns=['word', 'count'])
sns.barplot(x='word', y='count', data=df_common_words)
plt.xticks(rotation='vertical')
plt.show()


# In[133]:


df.head()


# In[134]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()


# In[136]:


x = cv.fit_transform(df['transformed_text']).toarray()


# In[137]:


x.shape


# In[138]:


y=df['target'].values


# In[139]:


y


# In[140]:


from sklearn.model_selection import train_test_split


# In[142]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[146]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score


# In[145]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[147]:


gnb.fit(x_train,y_train)
y_pred1 = gnb.predict(x_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[150]:


mnb.fit(x_train,y_train)
y_pred2 = mnb.predict(x_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[151]:


bnb.fit(x_train,y_train)
y_pred3 = bnb.predict(x_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[ ]:





# In[ ]:





# In[ ]:




