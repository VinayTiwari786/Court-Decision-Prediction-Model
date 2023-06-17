
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download("stopwords")
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import os
import string

from nltk.corpus import stopwords
stopword_list = stopwords.words('english')
for dirname, _, filenames in os.walk('justice.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('justice.csv', index_col=0)
pd.set_option('display.max_colwidth', None)
df.info()
df['name'].value_counts()
df.isnull().sum()
df.describe()
import_feature = ['ID', 'docket', 'term',
                  'facts', 'majority_vote', 'minority_vote',
                  'first_party_winner', 'decision_type']
df_n = df[import_feature]
df_n.isnull().sum()

df_n.dropna(inplace=True)

df_n.term.value_counts()
g = sns.countplot(x='decision_type', data=df_n)

g.set_xticklabels(g.get_xticklabels(), rotation=45)
plt.title("Decision Type Counts")
plt.xlabel("Decision Type")
plt.show()
sns.countplot(x='first_party_winner', data=df_n)
plt.show()


def remove_punctuations(text):
    translator = str.maketrans('', '', string.punctuation)
    return str(text).translate(translator)


def processrequest(requeststr):
    # remove repeated letters
    requeststr = remove_punctuations(requeststr)

    return requeststr

pd.options.mode.chained_assignment = None
df_n['facts'] = df_n['facts'].str.lower().apply(lambda x: processrequest(x))
df_n['facts'].head()
pro = preprocessing.LabelEncoder()
y = pro.fit_transform(df_n['first_party_winner'])
vect = TfidfVectorizer(min_df=0.0001, max_df=0.95, stop_words=stopword_list)
vect.fit(df_n.facts)
X = vect.transform(df_n.facts)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearSVC()
model.fit(X_train, y_train)


predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
model.fit(X_train, y_train)
model = DecisionTreeClassifier()