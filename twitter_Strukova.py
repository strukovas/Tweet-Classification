import tweepy           # To consume Twitter's API
import pandas as pd     # To handle data
import numpy as np      # For number computing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.utils import shuffle

# For plotting and visualization:
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Consume:
CONSUMER_KEY    = 'p7cpkuv0zPQaBd3X1jqIHnKn9'
CONSUMER_SECRET = '6APZspM93n5CLU9nvQSo3hSl5YevnZm2Bah3vGa38GjML5THED'

# Access:
ACCESS_TOKEN  = '1050397775572160512-9GvDeD9e9UIJVkkTBPYk3LApmLsqkk'
ACCESS_SECRET = 'qWoZt4Ul7g3FytwvP1pnkyXhFx8n3CgQNFvCyRbbMtsSt'


# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api

# We create an extractor object:
extractor = twitter_setup()
tweets1 = []
tweets2 = []
tweets3 = []
tweets4 = []

list_people = ['TheEllenShow','ConanOBrien','elonmusk','EmmaWatson']

for status in tweepy.Cursor(extractor.user_timeline, screen_name='TheEllenShow', tweet_mode="extended").items(500):
    tweets1.append(status.full_text)
    
for status in tweepy.Cursor(extractor.user_timeline, screen_name='ConanOBrien', tweet_mode="extended").items(500):
    tweets2.append(status.full_text)
    
for status in tweepy.Cursor(extractor.user_timeline, screen_name='elonmusk', tweet_mode="extended").items(500):
    tweets3.append(status.full_text)
    
for status in tweepy.Cursor(extractor.user_timeline, screen_name='EmmaWatson', tweet_mode="extended").items(500):
    tweets4.append(status.full_text)

list_tweets = [tweets1,tweets2,tweets3,tweets4]
 
print("Number of tweets extracted: "+str(len(tweets1))+","+str(len(tweets2))+","+str(len(tweets3))+","+str(len(tweets4))+".\n")

person1 = 1*np.ones(len(tweets1))
person2 = 2*np.ones(len(tweets2))
person3 = 3*np.ones(len(tweets3))
person4 = 4*np.ones(len(tweets4))

tweets1_train, tweets1_test, person1_train, person1_test = train_test_split(tweets1, person1, test_size=0.3, random_state=42)
tweets2_train, tweets2_test, person2_train, person2_test = train_test_split(tweets2, person2, test_size=0.3, random_state=42)
tweets3_train, tweets3_test, person3_train, person3_test = train_test_split(tweets3, person3, test_size=0.3, random_state=42)
tweets4_train, tweets4_test, person4_train, person4_test = train_test_split(tweets4, person4, test_size=0.3, random_state=42)

def analyse(tweets_train, tweets_test, person_train, person_test, classes):
    tweets_train, person_train = shuffle(tweets_train, person_train, random_state=42)
    tweets_test, person_test = shuffle(tweets_test, person_test, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english') 
    vectorizer.fit(tweets_train)
    vectorizer.fit(tweets_test)

    tfidf_train = vectorizer.transform(tweets_train)
    tfidf_test = vectorizer.transform(tweets_test)
    
    names = ['SVC, Linear Kernel','SVC, Poly Kernel','SVC, RBF Kernel','Naive Bayes']
    classifiers = [SVC(kernel='linear'),SVC(kernel='poly'),SVC(kernel='rbf'), GaussianNB()]
    
    accuracy = []

    for clf,name in zip(classifiers,names):
        clf.fit(tfidf_train.toarray(), person_train)

        y_test = person_test
        y_pred = clf.predict(tfidf_test.toarray())
        df = pd.DataFrame()
        df["actual"] = y_test
        df["predicted"] = y_pred
        correct = df[df["actual"] == df["predicted"]]

        accuracy.append(len(correct)/len(person_test))
        print(f"{name}: {len(correct)/len(person_test)*100:.2f}%") #to print percentage of correct predictions
        print()

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes, title=name)
        plt.show()


    model_names = ['Linear Kernel','Poly Kernel','RBF Kernel','Naive Bayes']
    n_models = len(model_names)

    plt.bar(np.arange(n_models),accuracy)
    plt.xticks(np.arange(n_models), model_names)
    plt.ylim(0,1);

tweets_train = np.concatenate((tweets1_train, tweets2_train))
tweets_test = np.concatenate((tweets1_test, tweets2_test))
person_train = np.concatenate((person1_train, person2_train))
person_test = np.concatenate((person1_test, person2_test))
analyse(tweets_train, tweets_test, person_train, person_test, classes=('TheEllenShow','ConanOBrien'))

tweets_train = np.concatenate((tweets3_train, tweets4_train))
tweets_test = np.concatenate((tweets3_test, tweets4_test))
person_train = np.concatenate((person3_train, person4_train))
person_test = np.concatenate((person3_test, person4_test))
analyse(tweets_train, tweets_test, person_train, person_test, classes=('elonmusk','EmmaWatson'))

tweets_train = np.concatenate((tweets1_train, tweets2_train, tweets3_train, tweets4_train))
tweets_test = np.concatenate((tweets1_test, tweets2_test, tweets3_test, tweets4_test))
person_train = np.concatenate((person1_train, person2_train, person3_train, person4_train))
person_test = np.concatenate((person1_test, person2_test, person3_test, person4_test))
analyse(tweets_train, tweets_test, person_train, person_test, classes=('TheEllenShow','ConanOBrien','elonmusk','EmmaWatson'))

def analyse(tweets_train, tweets_test, person_train, person_test, classes):
    tweets_train, person_train = shuffle(tweets_train, person_train, random_state=42)
    tweets_test, person_test = shuffle(tweets_test, person_test, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english') 
    vectorizer.fit(tweets_train)
    vectorizer.fit(tweets_test)

    tfidf_train = vectorizer.transform(tweets_train)
    tfidf_test = vectorizer.transform(tweets_test)
    
    names = ['SVC, Linear Kernel','SVC, Poly Kernel','SVC, RBF Kernel','Naive Bayes']
    classifiers = [SVC(kernel='linear'),SVC(kernel='poly'),SVC(kernel='rbf'), GaussianNB()]
    classifiers_improved = [SVC(kernel='linear', C=0.68),SVC(kernel='poly', gamma=0.99, degree=1),SVC(kernel='rbf', gamma=0.7), GaussianNB()]

    accuracy = []

    for clf,name in zip(classifiers_improved,names):
        clf.fit(tfidf_train.toarray(), person_train)

        y_test = person_test
        y_pred = clf.predict(tfidf_test.toarray())
        df = pd.DataFrame()
        df["actual"] = y_test
        df["predicted"] = y_pred
        correct = df[df["actual"] == df["predicted"]]

        accuracy.append(len(correct)/len(person_test))
        print(f"{name}: {len(correct)/len(person_test)*100:.2f}%") #to print percentage of correct predictions
        print()

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes, title=name)
        plt.show()


    model_names = ['Linear Kernel','Poly Kernel','RBF Kernel','Naive Bayes']
    n_models = len(model_names)

    plt.bar(np.arange(n_models),accuracy)
    plt.xticks(np.arange(n_models), model_names)
    plt.ylim(0,1);

tweets_train = np.concatenate((tweets1_train, tweets2_train))
tweets_test = np.concatenate((tweets1_test, tweets2_test))
person_train = np.concatenate((person1_train, person2_train))
person_test = np.concatenate((person1_test, person2_test))
analyse(tweets_train, tweets_test, person_train, person_test, classes=('TheEllenShow','ConanOBrien'))

tweets_train = np.concatenate((tweets3_train, tweets4_train))
tweets_test = np.concatenate((tweets3_test, tweets4_test))
person_train = np.concatenate((person3_train, person4_train))
person_test = np.concatenate((person3_test, person4_test))
analyse(tweets_train, tweets_test, person_train, person_test, classes=('elonmusk','EmmaWatson'))

tweets_train = np.concatenate((tweets1_train, tweets2_train, tweets3_train, tweets4_train))
tweets_test = np.concatenate((tweets1_test, tweets2_test, tweets3_test, tweets4_test))
person_train = np.concatenate((person1_train, person2_train, person3_train, person4_train))
person_test = np.concatenate((person1_test, person2_test, person3_test, person4_test))
analyse(tweets_train, tweets_test, person_train, person_test, classes=('TheEllenShow','ConanOBrien','elonmusk','EmmaWatson'))

pos = open('positive-words.txt', 'r')   
pos_words = pos.read().split()

neg = open('negative-words.txt', 'r')   
neg_words = neg.read().split()


for tweets, person in zip(list_tweets,list_people):
    n_positive = 0
    n_negative = 0
    n_words = 0
    for tweet in tweets:
        for word in tweet.split():
            n_words = n_words +1
            if word in pos_words:
                n_positive = n_positive +1
            elif word in neg_words:
                n_negative = n_negative +1

    percentage_positive = (n_positive/n_words)*100            
    print(f"{person}. Percentage of positive words: {percentage_positive:.2f}%")

    percentage_negative = (n_negative/n_words)*100            
    print(f"{person}. Percentage of negative words: {percentage_negative:.2f}%")
    print()
