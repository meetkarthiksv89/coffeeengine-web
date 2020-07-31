from flask import Flask, session, redirect, render_template, request, jsonify, flash, url_for
from flask_session import Session
from multiprocessing import Process
import json
import webbrowser
import jinja2



app = Flask(__name__)
app.config["SECRET_KEY"] = "new key"

# Configure session to use filesystem
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#import warnings
#warnings.filterwarnings("ignore", category=FutureWarning)
df = pd.read_csv('Coffee_Data_Set_Final_4.csv')
print(df)
df.shape
print(df.shape)
df.head(2).T
print(df.head(2).T)
# Create a new dataframe with two columns
# Create a new dataframe with two columns
df1 = df[['Product', 'Consumer complaint narrative']].copy()

# Remove missing values (NaN)
df1 = df1[pd.notnull(df1['Consumer complaint narrative'])]

# Renaming second column for a simpler name
df1.columns = ['Product', 'Consumer_complaint']

df1.shape
# Percentage of complaints with text
total = df1['Consumer_complaint'].notnull().sum()
round((total/len(df)*100),1)
pd.DataFrame(df.Product.unique()).values
# Because the computation is time consuming (in terms of CPU), the data was sampled
df2 = df1.sample(300, random_state=1).copy()
pd.DataFrame(df2.Product.unique())
print(pd.DataFrame(df2.Product.unique()))
# Create a new column 'category_id' with encoded categories
df2['category_id'] = df2['Product'].factorize()[0]
category_id_df = df2[['Product', 'category_id']].drop_duplicates()


# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Product']].values)

# New dataframe
df2.head()
print(df2.head)
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2),
                        stop_words='english')

# We transform each complaint into a vector
features = tfidf.fit_transform(df2.Consumer_complaint).toarray()

labels = df2.category_id

print("Each of the %d complaints is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))
# Finding the three most correlated terms with each of the product categories
N = 3
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("\n==> %s:" %(Product))
  print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
  print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))
  X = df2['Consumer_complaint']  # Collection of documents
  y = df2['Product']  # Target or the labels we want to predict (i.e., the 13 different complaints of products)

  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.25,
                                                      random_state=0)
models = [
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]

# 5 Cross-validation
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
std_accuracy = cv_df.groupby('model_name').accuracy.std()

acc = pd.concat([mean_accuracy, std_accuracy], axis= 1,
          ignore_index=True)
acc.columns = ['Mean Accuracy', 'Standard deviation']
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features,
                                                               labels,
                                                               df2.index, test_size=0.25,
                                                               random_state=1)
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('\t\t\t\tCLASSIFICATIION METRICS\n')
print(metrics.classification_report(y_test, y_pred,
                                    target_names= df2['Product'].unique()))

model.fit(features, labels)

N = 4
for Product, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("\n==> '{}':".format(Product))
  print("  * Top unigrams: %s" %(', '.join(unigrams)))
  print("  * Top bigrams: %s" %(', '.join(bigrams)))
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state = 0)

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2),
                        stop_words='english')

fitted_vectorizer = tfidf.fit(X_train)
tfidf_vectorizer_vectors = fitted_vectorizer.transform(X_train)


@app.route("/")
def index():
    """ Show search box """

    return render_template("question.html")

@app.route('/test', methods=['GET','POST'])
def test():
    data = request.get_json(force=False)
    print("First step accomplished")
    dataset=format(data)
    print(dataset , "step1")
    model = LinearSVC().fit(tfidf_vectorizer_vectors, y_train)
    print("got this far")
    new_complaint = dataset
    print(new_complaint, "step2")
    new_complaints = ''.join(new_complaint)
    print(new_complaints, "step3")
    values =model.predict(fitted_vectorizer.transform([new_complaints]))
    print(values, "step4")
    values2=" ".join(str(x) for x in values)
    labelling=str(values2)
    print(labelling, "step5")
    #
    # if labelling == "['Aroma Gold']":
    #         return(redirect("https://www.google.co.uk"))
    # elif labelling == "['Brown Gold']":
    #         return (redirect("https://www.google.co.uk"))
    #         # webbrowser.open_new_tab("https://pandurangacoffe.com/collections/frontpage/products/brown-gold")
    # elif labelling == "['French Blend']":
    #         return (redirect("https://www.google.co.uk"))
    # elif labelling == "['Grand Aroma']":
    #         return (redirect("https://www.google.co.uk"))
    print("Reached end")
    return render_template('response.html',labelling=labelling)






