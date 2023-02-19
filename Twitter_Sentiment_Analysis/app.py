import tweepy
import numpy as np
from textblob import TextBlob
from flask import Flask,render_template,request,redirect,session,jsonify,url_for
from cleantext import clean
app = Flask(__name__,static_folder="static")

consumer_key=''
consumer_secret=''

access_token=''
access_token_secret=''

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)
print(api)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analysis",methods=['POST'])
def tweet_analysis():
    if request.method == "POST":
        ques = request.form['ques']
    query= ques + "-filter:retweets"
    tweets = api.search_tweets(q=query,lang='en',count=10, result_type='popular')

    subjectivities = []
    polarities = []
    vals=[]
    print_result="none"
    for tweet in tweets:
        tweet.text=clean(tweet.text, no_emoji=True)
        phrase = TextBlob(tweet.text)
        if phrase.sentiment.polarity != 0.0 and phrase.sentiment.subjectivity != 0.0:
            polarities.append(phrase.sentiment.polarity)
            subjectivities.append(phrase.sentiment.subjectivity)
            vals.append(tweet.text)

        print('Tweet: ' + tweet.text)
        print('Polarity: ' + str(phrase.sentiment.polarity) + ' \ Subjectivity: ' + str(phrase.sentiment.subjectivity))
        print('.....................')

    result= {'polarity':polarities, 'subjectivity':subjectivities}

    get_weighted_polarity_mean= np.average(result['polarity'],weights=result['subjectivity'])

    get_polarity_mean = np.mean(result['polarity'])

    if get_polarity_mean > 0.0:
        print_result="POSITIVE"
    elif get_polarity_mean == 0.0:
        print_result="NEUTRAL"
    else:
        print_result="NEGATIVE"

    RES= {'polarity':polarities, 
    'subjectivity':subjectivities, 
    'values' : vals,
    'get_weighted_polarity_mean': get_weighted_polarity_mean,
    'get_polarity_mean':get_polarity_mean,
    'RESULT': print_result
    }
    # return render_template('result.html', ans=RES['RESULT'])
    return jsonify(RES)


if __name__ == "__main__":
     app.run(debug=True)


# POLARITY: It's a value from -1.0 to 1.0, where -1.0 referes to a 100% negative polarity and 1.0 to 100% positive polarity.

# SUBJECTIVITY: It's a value from 0.0 e 1.0, where 0 referes to a 100% objective text and 1.0 a 100% subjetive text.

# SUBJECTIVITY x OBJECTIVITY: Objective sentences usually contain facts and information, while subjective sentences express personal feelings and opinions.

# The text classification approach involves building classifiers from labeled instance