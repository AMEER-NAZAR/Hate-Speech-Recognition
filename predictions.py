import pandas as pd
import pickle
import re



with open("vectoriser.pkl", 'rb') as file:
    vect = pickle.load(file)
with open("random.pkl", 'rb') as file:
    random = pickle.load(file)
with open("NB.pkl", 'rb') as file:
    NBC = pickle.load(file)
with open("transformer.pkl", 'rb') as file:
    Trans = pickle.load(file)

def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())

def prediction_tweet(tweet):
    tweet_to_check = pd.DataFrame({"processed_tweets":tweet},index=[1])
    tweet_to_check["processed_tweets"] = tweet_to_check["processed_tweets"].apply(process_tweet)
    X = tweet_to_check['processed_tweets']
    x_test_counts = vect.transform(X)
    x_test_tfidf = Trans.transform(x_test_counts)
    prediction = NBC.predict(x_test_tfidf)
    if prediction == [0]:
        return "Not a hate speech"
    else:
        return "Hate Speech"
    



