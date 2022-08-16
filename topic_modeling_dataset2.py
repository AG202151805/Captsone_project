import requests
import os
import json
import pandas as pd
import csv
import datetime
import dateutil.parser
import unicodedata
import time
import numpy as np
import csv


df_2 = pd.read_csv('df_2.csv')

# topic modeling - creating set of words for professional and political topics

political_related_words = '''ביתר ערבי גזען ערבים גזענות המנון בהמנון ההמנון גזעני פופוליזם פופוליסט קפטן הקפטן
ביהודים ישראלי הישראלי המדינה מדינה מוסלמי מוסלמים המוסלמי המוסלמים במוסלמי במוסלמים כקפטן מלחמה בקפטן לקפטן
 גזע השמאל מיעוט מיעוטים טהורה ההימנון הימנון בהימנון משטרה המשטרה אלימות האלימות לגזענות לאומנות לאומני כיבוש
אלקין שמאלני סמולני ערבית מדינת אפרטהייד האפרטהייד פמיליה דייטש ליאון לאום הלאום חוק שותק שותקים פוליטית החוק
 דרוזים הדרוזים צרקסי צרקסים שמאלנית שמאלן השמאלנים שמאלנים התקווה בתקווה קיום כנסת הכנסת ירושלים בירושלים לירושלים
הערבי הערבים הגזען הגזענות בגזענות בגזען גיזען גיזענות גיזעני יהודי יהודים היהודי היהודים ביהודי צבא וההמנון ובהמנון
נבחרת בנבחרת הנבחרת פוליטיקה בפוליטיקה הפוליטיקה השמאלנית שחקנים השחקנים שחקן השחקן פשיזם הפשיזם דמוקרטי וההימנון
ובהימנון הכיבוש מסריח פלסטיני הפלסטיני פלסטינים הפלסטינים  ייצוג רקסי הדמוקרטי הדמוקרטית דמוקרטית דמוקרטיה הדמוקרטיה
 ערכים ערך ירדן ששר לביתר דמגוג דמגוגיה ישראל לישראל בישראל בביתר מקפטן דת דתו המוסלמית ברקוביץ שר שיר
אזרח אזרחות האזרחות האזרח שוויון שיוויון משוויון משיוויון בכנסת לכנסת השוויון השיוויון בשיוויון בשוויון לשיוויון
סמוטריץ שברקוביץ שליאון לליאון לברקוביץ מברקוביץ התגזען התגזענה המשואה משואה העצמאות עצמאות נולד סוגיית
בסוגיית עליהום העליהום שגליק גליק לגליק מגליק בגליק שואה בשואה לשואה השואה שרים יהדות שלום הר הבית בהר
להר ההר דתות בדתות הדת לדת בחירות הבחירות בבחירות זכויות הזכויות לזכויות זכות הזכות הצבא ביבי נתניהו סלפי הסלפי
בסלפי חמאר בוס הבוס בוסים הבוסים תמונה התמונה בתמונה לתמונה פילוג מחנאות הפילוג המחנאות פוסט הפוסט בפוסט
פוסטים הפוסטים בפוסטים מירי רגב לשויוון הממשלה טרור הטרור ממשלה בממשלה לממשלה בטרור המחבלים סמוטריץ מחבל תומכי
הליכוד ברקו גביר טיבי בבית בבית״ר בית״ר'''

sports_related_words = '''העונה דקות בהרכב שערים בליגה לשחק כבש שער שיחק בעונה
דקה בדקה בדקות שחק גול מאמן בקבוצה הקבוצה השער ליגה  השערים כובש כדור למשחק הרכב פנדל לחתום הוצע דיווחים תהליך
נקודות הכדורגל בנצחון הבלם המאמן בישול בישולים החלוץ בניצחון בשער גולים בגול בישל בכורה הבכורה בבכורה לבכורה
בהגנה במחצית הקשר לנצח לכבוש בכדור מחצית הכדור בקישור הכוכב בהפסד קבוצתו גמר רבע הגמר שלב בשלב הבתים חותם חתם
סיכם סיכמתי חתמתי ידיעה דיווח מועמדת המועמדת מובילה המובילה בלם בלמים מגן המגן שוער השוער מסירה נכנס
המסירה מסירות בעיטות המסירות חילוץ חילץ חילצה חילוצים ישחק אשחק המחצית מהמחצית לקבוצתו למעבר
 באנקר התקפי ההתקפה מהספסל ההרכבים פנדלים הקמפיין תארים הבעיטה התקפית הגנתית בפנדל תיקו יתרון מוצדק למונדיאל הגולים
עבירה העבירה הרמה קרן הפסידה יערוך נצחון שכובש החילופים השידור באימון ההחמצה לקישור נבחרת בנבחרת הנבחרת הגול
שחקנים השחקנים שחקן השחקן קשר חלוץ להפסיד בעט בעיטה החמיץ החמצה החטיא החטאה הפסד הבקיע להבקיע שדה צמד בגולים
חוזה להאריך באנגליה בספרד באיטליה אנגליה ספרד איטליה מוסקבה קריירה אוברייטד אובררייטד אנדרייטד אנדררייטד
בהגנה תפסיד פלייאוף ברצלונה מדריד צלסי ריאל יונייטד מנצסטר ליברפול אינטר יובנטוס מילאן פסז באיירן דורטמונד'''

political_related_words = political_related_words.replace('\n', '')
sports_related_words = sports_related_words.replace('\n', '')

# creating jaccard similarity functions
def jaccard_similarity(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_scores(group,tweets):
    scores = []
    for tweet in tweets:
        s = jaccard_similarity(group, tweet)
        scores.append(s)
    return scores

# get scores for each tweet
political_score = get_scores(political_related_words,df_2.clean_no_stops.to_list())
professional_score = get_scores(sports_related_words,df_2.clean_no_stops.to_list())

# create a jaccard score dataframe, wtih date, tweet, clean tweet and the topics score
data_scores  = {'Date':df_2.Date.to_list(), 'clean_no_stops':df_2.clean_no_stops.to_list(),
                'clean_tweet':df_2.clean_tweet.to_list(),
                'political_score':political_score,'professional_score': professional_score}

scores_df_2 = pd.DataFrame(data_scores)

# assign classes based on highest score
def get_classes(l1, l2):
    political = []
    professional = []

    for i, j in zip(l1, l2):
        m = max(i, j)
        if m == 0:
            professional.append(1)
        if m == i and m != 0:
            political.append(1)
        else:
            political.append(0)
        if m == j and m != 0:
            professional.append(1)
        else:
            professional.append(0)

    return political, professional


l1 = scores_df_2.political_score.to_list()
l2 = scores_df_2.professional_score.to_list()

# add category column for professional and political classification

scores_df_2['category'] = np.where(scores_df_2['political_score'] > scores_df_2['professional_score'], 'Political', 'Professional')

# exporting dataset in order to manually code a subset of the Tweets
scores_df_2.to_csv('scores_df2_full.csv')
