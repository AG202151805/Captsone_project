import os
import sys
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import string
from string import punctuation
import collections
from collections import Counter
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import KFold
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import warnings
warnings.filterwarnings('ignore')

# importing the file with the coded tweets
df_2_new = pd.read_csv('scores_df2_full.csv')


# new dataframe with relevant columns
df_2_new = df_2_new[['Date', 'clean_no_stops', 'sentiment code', 'category']].copy()

# subset with manually coded tweets
subset_2 = df_2_new[0:1350]

subset_2.dropna(inplace=True)
subset_2.reset_index(inplace=True)

# remove words with low frequency and tweets with small number of words

# split words into lists
v = subset_2['clean_no_stops'].str.split().tolist()
# compute global word frequency
c = Counter(chain.from_iterable(v))
# filter, join, and re-assign
subset_2['clean_new'] = [' '.join([j for j in i if c[j] > 3]) for i in v] # remove low frequency words

subset_2["count"] = ""
for i in range(len(subset_2)):
    subset_2['count'][i] = int(str(len(subset_2['clean_new'][i].split())))

subset_2 = subset_2[subset_2['count'] > 5] # keep tweets with more than 5 words

# manual stemming to dominant words in the dataset
subset_2['clean_new'] = (subset_2['clean_new'].str.replace('השחקן','שחקן').str.replace('השחקנים','שחקן')
                            .str.replace('כשחקן','שחקן').str.replace('שחקני','שחקן').str.replace('לשחקן','שחקן')
                            .str.replace('לשחקני','שחקן').str.replace('בשחקן','שחקן').str.replace('מהשחקנים','שחקן')
                            .str.replace('ושחקנים','שחקן').str.replace('והשחקן','שחקן').str.replace('ישחק','שחקן'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('ערבים','ערבי').str.replace('הערבי','ערבי')
                            .str.replace('הערבים','ערבי').str.replace('לערבים','ערבי').str.replace('והערבים','ערבי')
                            .str.replace('שהערבים','ערבי').str.replace('בערבית','ערבי').str.replace('ערביי','ערבי')
                            .str.replace('ערבית','ערבי').str.replace('וערבים','ערבי'))


subset_2['clean_new'] = (subset_2['clean_new'].str.replace('במשחק','משחק').str.replace('המשחק','משחק')
                            .str.replace('למשחק','משחק').str.replace('ומשחקים','משחק').str.replace('שמשחק','משחק')
                            .str.replace('המשחקים','משחק').str.replace('ומשחק','משחק').str.replace('מהמשחק','משחק')
                            .str.replace('שמשחקים','משחק').str.replace('משחקי','משחק').str.replace('למשחקים','משחק')
                           .str.replace('משחקת','משחק').str.replace('ומשחקת','משחק').str.replace('במשחקים','משחק'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('בנבחרת','נבחרת').str.replace('הנבחרת','נבחרת')
                            .str.replace('לנבחרת','נבחרת').str.replace('מהנבחרת','נבחרת').str.replace('שהנבחרת','נבחרת')
                            .str.replace('כשהנבחרת','נבחרת').str.replace('בנבחרות','נבחרת'))


subset_2['clean_new'] = (subset_2['clean_new'].str.replace('ישראלי','ישראל').str.replace('בישראל','ישראל')
                            .str.replace('הישראלי','ישראל').str.replace('לישראל','ישראל').str.replace('ישראלים','ישראל')
                            .str.replace('ישראלית','ישראל').str.replace('והישראלים','ישראל').str.replace('מהישראלים','ישראל')
                           .str.replace('הישראלים','ישראל').str.replace('שישראלי','ישראל').str.replace('כשישראלי','ישראל')
                           .str.replace('לישראלי','ישראל'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('שערים','שער').str.replace('השער','שער')
                            .str.replace('לשער','שער').str.replace('השערים','שער').str.replace('משערי','שער')
                            .str.replace('בשער','שער').str.replace('שלושער','שער').str.replace('משער','שער')
                           .str.replace('מהשער','שער').str.replace('שערי','שער').str.replace('ששער','שער')
                           .str.replace('ושער','שער'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('הקבוצה','קבוצה').str.replace('בקבוצה','קבוצה')
                            .str.replace('לקבוצה','קבוצה').str.replace('שהקבוצה','קבוצה').str.replace('וקבוצה','קבוצה'))


subset_2['clean_new'] = (subset_2['clean_new'].str.replace('דקות','דקה').str.replace('בדקה','דקה')
                            .str.replace('מדקה','דקה').str.replace('בדקות','דקה').str.replace('לדקה','דקה')
                            .str.replace('מהדקות','דקה'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('בעונה','עונה').str.replace('לעונה','עונה')
                            .str.replace('מהעונה','עונה').str.replace('מעונה','עונה').str.replace('לדקה','דקה')
                            .str.replace('מהדקות','דקה'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('במכבי','מכבי').str.replace('למכבי','מכבי')
                            .str.replace('ממכבי','מכבי').str.replace('ומכבי','מכבי').str.replace('שמכבי','מכבי')
                            .str.replace('מהדקות','דקה'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('במכבי','מכבי').str.replace('למכבי','מכבי')
                            .str.replace('ממכבי','מכבי').str.replace('ומכבי','מכבי').str.replace('שמכבי','מכבי'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('במקום','מקום').str.replace('למקום','מקום')
                            .str.replace('מקומות','מקום').str.replace('ובמקום','מקום').str.replace('מקומו','מקום')
                            .str.replace('במקומם','מקום').str.replace('ממקום','מקום').str.replace('במקומות','מקום'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('במקום','מקום').str.replace('למקום','מקום')
                            .str.replace('מקומות','מקום').str.replace('ובמקום','מקום').str.replace('מקומו','מקום')
                            .str.replace('במקומם','מקום').str.replace('ממקום','מקום').str.replace('במקומות','מקום'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('הליגה','בליגה').str.replace('לליגה','בליגה')
                            .str.replace('ליגה','בליגה').str.replace('הליגות','בליגה'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('ההרכב','הרכב').str.replace('בהרכב','הרכב')
                            .str.replace('ההרכבים','הרכב').str.replace('הרכבים','הרכב').str.replace('להרכב','הרכב'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('יהודי','יהודים').str.replace('יהודי','יהודים')
                            .str.replace('היהודים','יהודים').str.replace('היהודי','יהודים').str.replace('שיהודיה','יהודים')
                            .str.replace('ליהודים','יהודים').str.replace('מהיהודים','יהודים').str.replace('יהודית','יהודים')
                           .str.replace('יהודיות','יהודים'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('המנון','ההמנון').str.replace('ההימנון','ההמנון')
                            .str.replace('בהמנון','ההמנון').str.replace('בהימנון','ההמנון').str.replace('המנונים','ההמנון')
                            .str.replace('שבהמנון','ההמנון').str.replace('שבהימנון','ההמנון').str.replace('להמנון','ההמנון')
                           .str.replace('להימנון','ההמנון'))

subset_2['clean_new'] = (subset_2['clean_new'].str.replace('מוסלמים','מוסלמי').str.replace('המוסלמים','מוסלמי')
                            .str.replace('ממוסלמי','מוסלמי').str.replace('למוסלמי','מוסלמי').str.replace('למוסלמים','מוסלמי')
                            .str.replace('שמוסלמי','מוסלמי').str.replace('שמוסלמים','מוסלמי'))


subset_2['clean_new'] = (subset_2['clean_new'].str.replace('גזענית','גזעני').str.replace('גזענות','גזעני')
                            .str.replace('הגזענית','גזעני').str.replace('גזען','גזעני').str.replace('הגזען','גזעני')
                            .str.replace('גזעניים','גזעני').str.replace('הגזענות','גזעני').str.replace('מהגזענים','גזעני')
                           .str.replace('גזענים','גזעני').str.replace('הגזעניים','גזעני').str.replace('גזעי','גזעני')
                           .str.replace('וגזעי','גזעני').str.replace('גיזענית','גזעני').str.replace('גיזעני','גזעני')
                           .str.replace('הגיזענית','גזעני').str.replace('גיזען','גזעני').str.replace('גיזעני','גזעני')
                           .str.replace('גיזעניים','גזעני').str.replace('הגיזענות','גזעני').str.replace('מהגיזענים','גזעני')
                           .str.replace('גיזענים','גזעני').str.replace('הגיזעניים','גזעני').str.replace('גיזענות','גזעני'))

# sentiment analysis - training the models
def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

# fitting logistic regression model
tf_vector = get_feature_vector(np.array(subset_2.iloc[:, 5].values.astype('U')).ravel())
X = tf_vector.transform(np.array(subset_2.iloc[:, 5].values.astype('U')).ravel())
y = np.array(subset_2.iloc[:, 3]).ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

LM_model = LogisticRegressionCV(penalty='l2', max_iter=500, multi_class='multinomial', solver='lbfgs',
                                cv=10)
LM_model.fit(X_train,y_train)

# running the model on the rest of the tweets in the dataset
df2_rest = df_2_new[1350::]

df2_rest.drop('sentiment code', axis=1, inplace=True)
df2_rest.dropna(inplace=True)
df2_rest.reset_index(inplace=True)

# remove words with low frequency and tweets with small number of words

# split words into lists
v = df2_rest['clean_no_stops'].str.split().tolist()
# compute global word frequency
c = Counter(chain.from_iterable(v))
# filter, join, and re-assign
df2_rest['clean_new'] = [' '.join([j for j in i if c[j] > 3]) for i in v] # remove low frequency words

df2_rest["count"] = ""
for i in range(len(df2_rest)):
    df2_rest['count'][i] = int(str(len(df2_rest['clean_new'][i].split())))

df2_rest = df2_rest[df2_rest['count'] > 5] # keep tweets with more than 5 words

# manual stemming to dominant words in the dataset
df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('השחקן','שחקן').str.replace('השחקנים','שחקן')
                            .str.replace('כשחקן','שחקן').str.replace('שחקני','שחקן').str.replace('לשחקן','שחקן')
                            .str.replace('לשחקני','שחקן').str.replace('בשחקן','שחקן').str.replace('מהשחקנים','שחקן')
                            .str.replace('ושחקנים','שחקן').str.replace('והשחקן','שחקן').str.replace('ישחק','שחקן'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('ערבים','ערבי').str.replace('הערבי','ערבי')
                            .str.replace('הערבים','ערבי').str.replace('לערבים','ערבי').str.replace('והערבים','ערבי')
                            .str.replace('שהערבים','ערבי').str.replace('בערבית','ערבי').str.replace('ערביי','ערבי')
                            .str.replace('ערבית','ערבי').str.replace('וערבים','ערבי'))


df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('במשחק','משחק').str.replace('המשחק','משחק')
                            .str.replace('למשחק','משחק').str.replace('ומשחקים','משחק').str.replace('שמשחק','משחק')
                            .str.replace('המשחקים','משחק').str.replace('ומשחק','משחק').str.replace('מהמשחק','משחק')
                            .str.replace('שמשחקים','משחק').str.replace('משחקי','משחק').str.replace('למשחקים','משחק')
                           .str.replace('משחקת','משחק').str.replace('ומשחקת','משחק').str.replace('במשחקים','משחק'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('בנבחרת','נבחרת').str.replace('הנבחרת','נבחרת')
                            .str.replace('לנבחרת','נבחרת').str.replace('מהנבחרת','נבחרת').str.replace('שהנבחרת','נבחרת')
                            .str.replace('כשהנבחרת','נבחרת').str.replace('בנבחרות','נבחרת'))


df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('ישראלי','ישראל').str.replace('בישראל','ישראל')
                            .str.replace('הישראלי','ישראל').str.replace('לישראל','ישראל').str.replace('ישראלים','ישראל')
                            .str.replace('ישראלית','ישראל').str.replace('והישראלים','ישראל').str.replace('מהישראלים','ישראל')
                           .str.replace('הישראלים','ישראל').str.replace('שישראלי','ישראל').str.replace('כשישראלי','ישראל')
                           .str.replace('לישראלי','ישראל'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('שערים','שער').str.replace('השער','שער')
                            .str.replace('לשער','שער').str.replace('השערים','שער').str.replace('משערי','שער')
                            .str.replace('בשער','שער').str.replace('שלושער','שער').str.replace('משער','שער')
                           .str.replace('מהשער','שער').str.replace('שערי','שער').str.replace('ששער','שער')
                           .str.replace('ושער','שער'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('הקבוצה','קבוצה').str.replace('בקבוצה','קבוצה')
                            .str.replace('לקבוצה','קבוצה').str.replace('שהקבוצה','קבוצה').str.replace('וקבוצה','קבוצה'))


df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('דקות','דקה').str.replace('בדקה','דקה')
                            .str.replace('מדקה','דקה').str.replace('בדקות','דקה').str.replace('לדקה','דקה')
                            .str.replace('מהדקות','דקה'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('בעונה','עונה').str.replace('לעונה','עונה')
                            .str.replace('מהעונה','עונה').str.replace('מעונה','עונה').str.replace('לדקה','דקה')
                            .str.replace('מהדקות','דקה'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('במכבי','מכבי').str.replace('למכבי','מכבי')
                            .str.replace('ממכבי','מכבי').str.replace('ומכבי','מכבי').str.replace('שמכבי','מכבי')
                            .str.replace('מהדקות','דקה'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('במכבי','מכבי').str.replace('למכבי','מכבי')
                            .str.replace('ממכבי','מכבי').str.replace('ומכבי','מכבי').str.replace('שמכבי','מכבי'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('במקום','מקום').str.replace('למקום','מקום')
                            .str.replace('מקומות','מקום').str.replace('ובמקום','מקום').str.replace('מקומו','מקום')
                            .str.replace('במקומם','מקום').str.replace('ממקום','מקום').str.replace('במקומות','מקום'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('במקום','מקום').str.replace('למקום','מקום')
                            .str.replace('מקומות','מקום').str.replace('ובמקום','מקום').str.replace('מקומו','מקום')
                            .str.replace('במקומם','מקום').str.replace('ממקום','מקום').str.replace('במקומות','מקום'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('הליגה','בליגה').str.replace('לליגה','בליגה')
                            .str.replace('ליגה','בליגה').str.replace('הליגות','בליגה'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('ההרכב','הרכב').str.replace('בהרכב','הרכב')
                            .str.replace('ההרכבים','הרכב').str.replace('הרכבים','הרכב').str.replace('להרכב','הרכב'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('יהודי','יהודים').str.replace('יהודי','יהודים')
                            .str.replace('היהודים','יהודים').str.replace('היהודי','יהודים').str.replace('שיהודיה','יהודים')
                            .str.replace('ליהודים','יהודים').str.replace('מהיהודים','יהודים').str.replace('יהודית','יהודים')
                           .str.replace('יהודיות','יהודים'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('המנון','ההמנון').str.replace('ההימנון','ההמנון')
                            .str.replace('בהמנון','ההמנון').str.replace('בהימנון','ההמנון').str.replace('המנונים','ההמנון')
                            .str.replace('שבהמנון','ההמנון').str.replace('שבהימנון','ההמנון').str.replace('להמנון','ההמנון')
                           .str.replace('להימנון','ההמנון'))

df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('מוסלמים','מוסלמי').str.replace('המוסלמים','מוסלמי')
                            .str.replace('ממוסלמי','מוסלמי').str.replace('למוסלמי','מוסלמי').str.replace('למוסלמים','מוסלמי')
                            .str.replace('שמוסלמי','מוסלמי').str.replace('שמוסלמים','מוסלמי'))


df2_rest['clean_new'] = (df2_rest['clean_new'].str.replace('גזענית','גזעני').str.replace('גזענות','גזעני')
                            .str.replace('הגזענית','גזעני').str.replace('גזען','גזעני').str.replace('הגזען','גזעני')
                            .str.replace('גזעניים','גזעני').str.replace('הגזענות','גזעני').str.replace('מהגזענים','גזעני')
                           .str.replace('גזענים','גזעני').str.replace('הגזעניים','גזעני').str.replace('גזעי','גזעני')
                           .str.replace('וגזעי','גזעני').str.replace('גיזענית','גזעני').str.replace('גיזעני','גזעני')
                           .str.replace('הגיזענית','גזעני').str.replace('גיזען','גזעני').str.replace('גיזעני','גזעני')
                           .str.replace('גיזעניים','גזעני').str.replace('הגיזענות','גזעני').str.replace('מהגיזענים','גזעני')
                           .str.replace('גיזענים','גזעני').str.replace('הגיזעניים','גזעני').str.replace('גיזענות','גזעני'))


# predict the sentiment of the Tweets
df2_rest_X = tf_vector.transform(np.array(df2_rest.iloc[:, 4].values.astype('U')).ravel())

# apply the whole pipeline to data
df2_pred_rest = pd.DataFrame(LM_model.predict(df2_rest_X))

# reset index of df2_rest in order to merge with df2_pred_rest
df2_rest.reset_index(inplace=True)

# remove level_0 column
df2_rest.drop('level_0', axis=1, inplace=True)

# join the rest dataframe with the pred_rest predictions
df2_rest_full = df2_rest.join(df2_pred_rest)

# rename sentiment code column
df2_rest_full.rename(columns = {0:'sentiment code'}, inplace = True)

# merge the two dataframes subset_2 and df2_rest_full
subset_2 = subset_2[['index', 'Date', 'clean_no_stops', 'category', 'clean_new', 'count', 'sentiment code']]

frames = [subset_2, df2_rest_full]

df2_all_full = pd.concat(frames)

df2_all_full.reset_index(inplace=True)
df2_all_full.drop('level_0', axis=1, inplace=True)

# assign sentiment column based on sentiment code
def calc_new_col(row):
    if row['sentiment code'] == 2.0:
        return 'Positive'
    elif row['sentiment code'] == 1.0:
        return 'Neutral'
    else:
        return 'Negative'

df2_all_full["sentiment"] = df2_all_full.apply(calc_new_col, axis=1)

# sort dataframe by date
df2_all_full.sort_values(by='Date', inplace=True)

# change date column to date format
df2_all_full['Date'] = df2_all_full['Date'].astype('datetime64[ns]')

# Create dataframe with category and sentiment as columns for each class
df2_all_class = df2_all_full[['Date', 'sentiment', 'category']].copy()

# create columns for each sentiment and topic
df2_all_class["Negative"] = 0
df2_all_class["Neutral"] = 0
df2_all_class["Positive"] = 0
df2_all_class["Political"] = 0
df2_all_class["Professional"] = 0

# count the sentiment and topic in each row
for i in range(len(df2_all_class['sentiment'])):
    if df2_all_class['sentiment'][i] == 'Negative':
        df2_all_class['Negative'][i] = 1
    elif df2_all_class['sentiment'][i] == 'Neutral':
        df2_all_class['Neutral'][i] = 1
    else:
        df2_all_class['Positive'][i] = 1

for i in range(len(df2_all_class['category'])):
    if df2_all_class['category'][i] == 'Political':
        df2_all_class['Political'][i] = 1
    else:
        df2_all_class['Professional'][i] = 1

# count the combination of sentiment and topic in each row
df2_all_class["Negative Political"] = 0
df2_all_class["Negative Professional"] = 0
df2_all_class["Neutral Political"] = 0
df2_all_class["Neutral Professional"] = 0
df2_all_class["Positive Political"] = 0
df2_all_class["Positive Professional"] = 0

for i in range(len(df2_all_class['sentiment'])):
    if df2_all_class['Negative'][i] == 1 and df2_all_class['Political'][i] == 1:
        df2_all_class['Negative Political'][i] = 1
    elif df2_all_class['Negative'][i] == 1 and df2_all_class['Professional'][i] == 1:
        df2_all_class['Negative Professional'][i] = 1

    elif df2_all_class['Neutral'][i] == 1 and df2_all_class['Political'][i] == 1:
        df2_all_class['Neutral Political'][i] = 1
    elif df2_all_class['Neutral'][i] == 1 and df2_all_class['Professional'][i] == 1:
        df2_all_class['Neutral Professional'][i] = 1

    elif df2_all_class['Positive'][i] == 1 and df2_all_class['Political'][i] == 1:
        df2_all_class['Positive Political'][i] = 1
    elif df2_all_class['Positive'][i] == 1 and df2_all_class['Professional'][i] == 1:
        df2_all_class['Positive Professional'][i] = 1

# removing the sentiment and category columns and aggregating by date each class
df2_all_daily = df2_all_class.groupby('Date').agg({'Negative':'sum', 'Neutral':'sum', 'Positive':'sum',
                         'Political':'sum', 'Professional': 'sum', 'Negative Political':'sum',
                        'Negative Professional': 'sum', 'Neutral Political': 'sum',
                        'Neutral Professional': 'sum', 'Positive Political': 'sum',
                                        'Positive Professional': 'sum'}).reset_index()

# transposing the dataframe so dates are as columns, in order to convert the data from daily to weekly
# making the date row the columns and removing the row
df2_all_t = df2_all_daily.transpose()
df2_all_t.columns = df2_all_t.iloc[0]
df2_all_t.drop(df2_all_t.head(1).index,inplace=True)


# converting to weekly aggregation

def new_case_count(state_new_cases):
    first_Monday_found = False
    week_case_count = 0
    week_case_counts = []
    for index, value in state_new_cases.items():
        index_date = pd.to_datetime(index, format='%Y/%m/%d',
                                    exact = False)
        index_day_of_week = index_date.day_name()
        if not first_Monday_found and index_day_of_week != 'Monday':
            continue
        first_Monday_found = True
        week_case_count += value
        if index_day_of_week == 'Sunday':
            week_case_counts.append(week_case_count)
            week_case_count = 0
    return week_case_counts

# converting list to DataFrame object
df2_all_weekly =  pd.DataFrame(new_case_count(df2_all_t))

df2_all_weekly.reset_index(inplace=True)

# rename sentiment code column
df2_all_weekly.rename(columns = {'index':'Date'}, inplace = True)

# Creating transposed daily and monthly dataframe

df2_daily_final = df2_all_t.transpose().reset_index()
df2_daily_final.set_index('Date', inplace=True)
df2_daily_final.index = pd.to_datetime(df2_daily_final.index)

df2_all_monthly = df2_daily_final.resample('1M').sum()

# Reset index so date is the first column in both daily and monthly
df2_daily_final.reset_index(inplace = True)
df2_all_monthly.reset_index(inplace = True)
