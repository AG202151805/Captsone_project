import requests
import os
import json
import pandas as pd
import csv
import datetime
import dateutil.parser
import unicodedata
import time
import tweepy
import numpy as np
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import gensim.utils
from gensim.utils import simple_preprocess

os.environ['TOKEN'] = 'Bearer Token'

# Now, we will create our auth() function, which retrieves the token from the environment.

def auth():
    return os.getenv('TOKEN')

# Next, we will define a function that will take our bearer token, pass it for
# authorization and return headers we will use to access the API.

def create_headers(bearer_token):
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    return headers

# Now that we can access the API, we will build the request for the endpoint
# we are going to use and the parameters we want to pass.

def create_url(keyword, start_date, end_date, max_results = 10):

    search_url = "https://api.twitter.com/2/tweets/search/all" #Change to the endpoint you want to collect data from

    #change params based on the endpoint you are using
    query_params = {'query': keyword,
                    'start_time': start_date,
                    'end_time': end_date,
                    'max_results': max_results,
                    'expansions': 'author_id,in_reply_to_user_id,geo.place_id',
                    'tweet.fields': 'id,text,author_id,in_reply_to_user_id,geo,conversation_id,created_at,lang,public_metrics,referenced_tweets,reply_settings,source',
                    'user.fields': 'id,name,username,created_at,description,public_metrics,verified',
                    'place.fields': 'full_name,id,country,country_code,geo,name,place_type',
                    'next_token': {}}
    return (search_url, query_params)



def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def append_to_csv(json_response, fileName):

    #A counter variable
    counter = 0

    #Open OR create the target CSV file
    csvFile = open(fileName, "a", newline="", encoding='utf-8')
    csvWriter = csv.writer(csvFile)

    #Loop through each tweet
    for tweet in json_response['data']:

        # We will create a variable for each since some of the keys might not exist for some tweets
        # So we will account for that

        # 1. Author ID
        author_id = tweet['author_id']

        # 2. Time created
        created_at = dateutil.parser.parse(tweet['created_at'])

        # 3. Geolocation
        if ('geo' in tweet):
            geo = tweet['geo']['place_id']
        else:
            geo = " "

        # 4. Tweet ID
        tweet_id = tweet['id']

        # 5. Conversation ID
        conversation_id = tweet['conversation_id']

        # 6. Language
        lang = tweet['lang']


        # 7. source
        source = tweet['source']

        # 8. Tweet text
        text = tweet['text']

        # Assemble all data in a list
        res = [author_id, created_at, geo, tweet_id, conversation_id, lang, source, text]

        # Append the result to the CSV file
        csvWriter.writerow(res)
        counter += 1

    # When done, close the CSV file
    csvFile.close()

    # Print the number of tweets for this iteration
    print("# of Tweets added from this response: ", counter)

#Inputs for tweets
bearer_token = auth()
headers = create_headers(bearer_token)
keyword = ["נתכו lang:he", "נאתכו lang:he", "סבע lang:he", "דאבור lang:he",
           "טוואטחה lang:he", "טווטחה lang:he", "טאהא lang:he", "טהאא lang:he"
          , "חבשי lang:he", "כיאל lang:he", "בירם lang:he", "אלחמיד lang:he"
          , "אל חמיד lang:he", "ראדי lang:he", "שחקנים מוסלמים lang:he", "שחקן מוסלמי lang:he",
           "מוסלמי בנבחרת lang:he","מוסלמים בנבחרת lang:he", "נבחרת מוסלמים lang:he",
           "השחקנים המוסלמים lang:he", "השחקן המוסלמי lang:he",
          "שחקנים ערבים lang:he", "שחקן ערבי lang:he", "ערבי בנבחרת lang:he",
          "ערבים בנבחרת lang:he", "נבחרת ערבים lang:he", "השחקנים הערבים lang:he",
          "השחקן הערבי lang:he", "שחקן כדורגל ערבי lang:he"]

start_list =    ['2017-09-15T00:00:00.000Z',
                 '2017-10-01T00:00:00.000Z',
                 '2017-11-01T00:00:00.000Z',
                 '2017-12-01T00:00:00.000Z',
                 '2018-01-01T00:00:00.000Z',
                 '2018-02-01T00:00:00.000Z',
                 '2018-03-01T00:00:00.000Z',
                 '2018-04-01T00:00:00.000Z',
                 '2018-05-01T00:00:00.000Z',
                 '2018-06-01T00:00:00.000Z',
                 '2018-07-01T00:00:00.000Z',
                 '2018-08-01T00:00:00.000Z',
                 '2018-09-01T00:00:00.000Z',
                 '2018-10-01T00:00:00.000Z',
                 '2018-11-01T00:00:00.000Z',
                 '2018-12-01T00:00:00.000Z',
                 '2019-01-01T00:00:00.000Z',
                 '2019-02-01T00:00:00.000Z',
                 '2019-03-01T00:00:00.000Z',
                 '2019-04-01T00:00:00.000Z',
                 '2019-05-01T00:00:00.000Z',
                 '2019-06-01T00:00:00.000Z',]

end_list =      ['2017-09-30T23:59:59.000Z',
                 '2017-10-31T23:59:59.000Z',
                 '2017-11-30T23:59:59.000Z',
                 '2017-12-31T23:59:59.000Z',
                 '2018-01-31T23:59:59.000Z',
                 '2018-02-28T23:59:59.000Z',
                 '2018-03-31T23:59:59.000Z',
                 '2018-04-30T23:59:59.000Z',
                 '2018-05-31T23:59:59.000Z',
                 '2018-06-30T23:59:59.000Z',
                 '2018-07-31T23:59:59.000Z',
                 '2018-08-31T23:59:59.000Z',
                 '2018-09-30T23:59:59.000Z',
                 '2018-10-31T23:59:59.000Z',
                 '2018-11-30T23:59:59.000Z',
                 '2018-12-31T23:59:59.000Z',
                 '2019-01-31T23:59:59.000Z',
                 '2019-02-28T23:59:59.000Z',
                 '2019-03-31T23:59:59.000Z',
                 '2019-04-30T23:59:59.000Z',
                 '2019-05-31T23:59:59.000Z',
                 '2019-06-30T23:59:59.000Z']
max_results = 500

#Total number of tweets we collected from the loop
total_tweets = 0

# Create file
csvFile = open("data_set_1.csv", "a", newline="", encoding='utf-8')
csvWriter = csv.writer(csvFile)

#Create headers for the data you want to save, in this example, we only want save these columns in our dataset
csvWriter.writerow(['author id', 'created_at', 'geo', 'id', 'conversation_id', 'lang', 'source','tweet'])
csvFile.close()

for j in range(0,len(keyword)):
    for i in range(0,len(start_list)):

        # Inputs
        count = 0 # Counting tweets per time period
        max_count = 10000 # Max tweets per time period
        flag = True
        next_token = None

        # Check if flag is true
        while flag:
            # Check if max_count reached
            if count >= max_count:
                break
            print("-------------------")
            print("Token: ", next_token)
            url = create_url(keyword[j], start_list[i],end_list[i], max_results)
            json_response = connect_to_endpoint(url[0], headers, url[1], next_token)
            result_count = json_response['meta']['result_count']

            if 'next_token' in json_response['meta']:
                # Save the token to use for next call
                next_token = json_response['meta']['next_token']
                print("Next Token: ", next_token)
                if result_count is not None and result_count > 0 and next_token is not None:
                    print("Start Date: ", start_list[i])
                    append_to_csv(json_response, "data_set_1.csv")
                    count += result_count
                    total_tweets += result_count
                    print("Total # of Tweets added: ", total_tweets)
                    print("-------------------")
                    time.sleep(5)
            # If no next token exists
            else:
                if result_count is not None and result_count > 0:
                    print("-------------------")
                    print("Start Date: ", start_list[i])
                    append_to_csv(json_response, "data_set_1.csv")
                    count += result_count
                    total_tweets += result_count
                    print("Total # of Tweets added: ", total_tweets)
                    print("-------------------")
                    time.sleep(5)

                #Since this is the final request, turn flag to false to move to the next time period.
                flag = False
                next_token = None
            time.sleep(5)
    print("Total number of results: ", total_tweets)


df_1 = pd.read_csv('data_set_1.csv')

# removing links, usernames, retweets etc.

def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'http\S+', '', tweet) # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet) # rempve bitly links
    tweet = tweet.strip('[link]') # remove [links]
    return tweet

def remove_users(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove retweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet) # remove tweeted at
    return tweet

my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem

# cleaning master function
def clean_tweet(tweet, bigrams=False):
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = tweet.lower() # lower case
    tweet = re.sub('['+my_punctuation + ']+', ' ', tweet) # strip punctuation
    tweet = re.sub('\s+', ' ', tweet) #remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet) # remove numbers
    tweet = re.sub(r'@\w+',"",tweet)
    tweet = re.sub(r'#\w+',"",tweet)
    tweet = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet)
    tweet_token_list = [word for word in tweet.split(' ')] # remove stopwords

    tweet_token_list = [word_rooter(word) if '#' not in word else word
                        for word in tweet_token_list] # apply word rooter
    if bigrams:
        tweet_token_list = tweet_token_list+[tweet_token_list[i]+'_'+tweet_token_list[i+1]
                                            for i in range(len(tweet_token_list)-1)]
    tweet = ' '.join(tweet_token_list)
    return tweet

# stopswords in Hebrew
stopwords_he = ['אני', 'את','אתה','אנחנו','אתן','אתם','הם','הן','היא','הוא','שלי','שלו','שלך','שלה','שלנו','שלכם',
'שלכן','שלהם','שלהן','לי','לו','לה','לנו','לכם','לכן','להם','להן','אותה','אותו','זה','זאת','אלה','אלו','תחת','מתחת',
'מעל','בין','עם','עד','נגר','על','אל','מול','של','אצל','כמו','אחר','אותו','בלי','לפני','אחרי','מאחורי','עלי','עליו',
'עליה','עליך','עלינו','עליכם','עליכן','עליהם','עליהן','כל','כולם','כולן','כך','ככה','כזה','זה','זאת','אותי','אותה',
'אותם','אותך','אותו','אותן','אותנו','ואת','את','אתכם','אתכן','איתי','איתו','איתך','איתה','איתם','איתן','איתנו','איתכם',
'איתכן','יהיה','תהיה','היתי','היתה','היה','להיות','עצמי','עצמו','עצמה','עצמם','עצמן','עצמנו','עצמהם','עצמהן','מי','מה',
'איפה','היכן','במקום שבו','אם','לאן','למקום שבו','מקום בו','איזה','מהיכן','איך','כיצד','באיזו מידה','מתי','בשעה ש',
'כאשר','כש','למרות','לפני','אחרי','מאיזו סיבה','הסיבה שבגללה','למה','מדוע','לאיזו תכלית','כי','יש','אין','אך','מנין','מאין',
'מאיפה','יכל','יכלה','יכלו','יכול','יכולה','יכולים','יכולות','יוכלו','יוכל','מסוגל','לא','רק','אולי','אין','לאו','אי',
'כלל','נגד','אם','עם','אל','אלה','אלו','אף','על','מעל','מתחת','מצד','בשביל','לבין','באמצע','בתוך','דרך','מבעד','באמצעות',
'למעלה','למטה','מחוץ','מן','לעבר','מכאן','כאן','הנה','הרי','פה','שם','אך','ברם','שוב','אבל','מבלי','בלי','מלבד','רק',
'בגלל','מכיוון','עד','אשר','ואילו','למרות','אס','כמו','כפי','אז','אחרי','כן','לכן','לפיכך','מאד','עז','בו',
'מעט','מעטים','במידה','שוב','יותר','מדי','גם','כן','נו','אחר','אחרת','אחרים','אחרות','אשר','או']

df_1['clean_tweet'] = df_1.tweet.apply(clean_tweet)

def remove_stopwords(tweets):
    return [[word for word in simple_preprocess(tweet) if word  not in stopwords_he] for tweet in tweets]
df_1['no_token_stop'] = remove_stopwords(df_1['clean_tweet'])


# remove duplicates
df_1 = df_1.drop_duplicates(subset=['clean_tweet'], keep='first')

# remove retweets
df_1 = df_1[df_1["clean_tweet"].str.contains("rt") == False]

# remove stopwords
pat = r'\b(?:{})\b'.format('|'.join(stopwords_he))
df_1['clean_no_stops'] = df_1['clean_tweet'].str.replace(pat, '')

# remove emoji
df_1['clean_no_stops'] = df_1['clean_no_stops'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)

#remove \n
df_1.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)

# export dataframe
df_1.to_csv('df_1.csv')
