---
layout: post
title: Extracting keywords from text
tags: nlp tf-idf machine-learning classification
---

Keywords are used in several fields as a way to summarize and categorize natural text, such as papers, essays, news articles or blog posts. Keywords, or tags, allow a quick and easy categorization of available information, thus facilitating its access and search. However, while many of existing natural texts have been assiegned their own keywords, either by their authors or by third parties, the vast majority of the information out there is currently untagged and, in many cases, virtually inaccessible. This begs the question: how difficult is it to develop a system that performs automatic and reliable tagging of written documents?

This post is a quick and dirty attempt to provide an answer to the above question. While the approach presented here is certainly not the most sophisticated, it delivers some encouraging preliminary results and provides a benchmark against which more complex solutions can be compared.

The approach outlined in this document consists of the following steps:

  - Overview
  - Data Loading and Preprocessing
  - Metric Definition
  - Natural Text Processing
    - Defining the Keywords Set
    - Building the Vocabulary
    - Building the TF-IDF matrix
  - Fit and Predict
    - Creating a Multi Label Binarizer
    - Building the Classifier
  - Results
    - Using the Title
    - Using the Body
  - Discussion
  - Appendix

The post is intended as a walkthrough of the methodology and provides a complete codebase to implement your own model.


## Overview

This effort is aimed at extracting a set of keywords from individual documents extracted from the "Ask Ubuntu" database, freely available on <a href="https://stackexchange.com/">Stack Exchange</a>, specifically its subset questions collected from January 2013 to June 2017(*). The dataset comes with a set of fields, out of which only the title, body and keywords are used for this analysis.

The implementation pipeline is fairly simple. The title and body of each document undergo a simple NLP preprocessing (e.g. standard stop words, punctuation), then title and body TF-IDF matrices are produced and fed to a One-Vs-Rest classifier, which outputs, for each document, a list of probabilities, each corresponding to one individual keyword. Using a probability threshold - which can also be learned from data - each document can be ultimately assigned a set of keywords.

The performance of the model is evaluated using the <a href="https://www.kaggle.com/wiki/MeanFScore">Mean-F-Score</a>.

(*) Note: Due to computational limitations, the dataset is further reduced in the analysis presented here.

## Data Loading and Preprocessing

Data processing consists of dataset loading (from a set of files previously downloaded via Stack Exchange query) and train-test splitting. For the latter, the data from January 1st, 2017 to June, 30th, 2017, is used as test set. This choice, as opposed to a random split, is motivated by the fact that it is considered good practice to use the most recent data as test set for application based on time stamped data.


```python
def process_source_df(s_df):
    cols = ['Id', 'UserId', 'Title', 'Body', 'Tags', 'CreationDate']
    s_df = s_df[cols]
    s_df['CreationDate'] = pd.to_datetime(s_df['CreationDate'])
    s_df = s_df[np.isnan(s_df['UserId'])==False]
    s_df['UserId'] = s_df['UserId'].astype(int)
    s_df['Tags'] = s_df['Tags'].apply(lambda tag: re.split('><', str(tag).strip('><')))    
    return s_df
#end 

def read_input():
    pref = 'au_'
    for sid in range(1, 11):
        fname = source_path + pref + str(sid).zfill(2) + '.csv'
        s_df = pd.read_csv(fname)
        s_df = process_source_df(s_df)
        if sid==1:
            df = s_df
        else:
            df = pd.concat([df, s_df]).reset_index(drop=True)
        #end
    #end
    return df.dropna(how='any')
#end
```

## Metric Definition

It is good measure to start developing any algorithm with a solid idea of how to assess its performance. For this reason, I am also defining the performance metric before I dive into the description of the algorithm used. As mentioned above, I use the Mean-F-Score, as follows.

```python
def f_scoring(true_ls, pred_ls):
    n_matches = len(set(true_ls) & set(pred_ls))
    p = 1.0 * n_matches / len(true_ls)
    r = 1.0 * n_matches / len(pred_ls)
    if (p + r)==0:
        f_score = 0
    else:
        f_score = 2* (p * r) / (p + r)
    #end
    return f_score
#end
    
def mean_f_scoring(true_ls_ls, pred_ls_ls):
    N = 1.0 * len(true_ls_ls)
    mean_f_score = 0.0
    for t_ls, p_ls in zip(true_ls_ls, pred_ls_ls):
        mean_f_score += f_scoring(t_ls, p_ls) / N
    #end
    return mean_f_score
#end
```

## Natural Text Processing

### Defining the Keywords Set

Every document will be potentially assigned a set of keywords taken as the list of unique keywords present in the train set.

```python
def get_unique_kw(df):
    kw_ls = list()
    kw_ls_ls = list(df['Tags'])
    for kwls in kw_ls_ls:
        kw_ls += kwls
    #end
    return list(set(kw_ls))
#end
```

### Building the Vocabulary

To procuce the TF-IDF matrices, one has to define a vocabulary. This, in turns, requires a corpus onto which to train. I here use the train set as corpus, define a CountVectorizer, using the default scikit-learn class, and extract the corresponding vocabulary using the english stopwords (from NLTK) plus punctuation. A filter on bi-grams and on the size of the vocabulary. Specifically, the common words, as well as the least common words are neglected. The thesholds are arbitrary (for example, words that appear on more than 60% of the titles are neglected), but could in principle be learned from data as well.

```python
def build_vocabulary(df, field='Body'):
    stop_words = stopwords.words('english') + list(punctuation)
    if field=='Body':
        vect = CountVectorizer(stop_words=stop_words, min_df=0.01, max_df=0.5, ngram_range=(1,2))
    elif field=='Title':
        vect = CountVectorizer(stop_words=stop_words, min_df=0.001, max_df=0.6, ngram_range=(1,2))
    #end
    vf = vect.fit(df[field])
    vocab_fd_ls = vect.vocabulary_.keys()
    vocab_kw_ls = get_unique_kw(df)
    vocabulary = list(set(vocab_fd_ls + vocab_kw_ls))
    return vocabulary
#end
```

### Building the TF-IDF matrix

Using the vocabulary and the text, the TF-IDF matrices are quickly built as follows.

```python
def create_tfidf_smx(trn_text, tst_text, vocab):
    vect = CountVectorizer(vocabulary=vocab)
    freq_trn = vect.fit_transform(trn_text)
    freq_tst = vect.transform(tst_text)
    tfidf = TfidfTransformer()
    tfifd_trn_smx = tfidf.fit_transform(freq_trn)
    tfifd_tst_smx = tfidf.transform(freq_tst)
    return tfifd_trn_smx, tfifd_tst_smx
#end
```

## Fit and Predict

### Creating a Multi Label Binarizer

This section is necessary to create a multi label binarizer that turns the list of keywords assigned to each document into a vector of 0's and 1's. This step, which requires the sequential application of Label Encoding and Multi Label Binarizer, is necessary to create the target binary vectors to be fed to the fit method of the classifier.

```python
def build_le(train_df):
    LE = LabelEncoder()
    LE.fit(get_unique_kw(train_df) + ['other'])
    return LE
#end

def build_le_mlb_trg(train_df):
    LE = build_le(train_df)
    target_lb = [LE.transform(tags).tolist() for tags in list(train_df['Tags'])]
    MLB = MultiLabelBinarizer()
    target_mlb = MLB.fit_transform(target_lb)
    return target_mlb, LE, MLB
#end

def ml_binarize(LE, MLB, kw_ls_ls):
    return MLB.transform(LE.transform(kw_ls_ls))
#end

def ml_invert_binarize(LE, MLB, mlb_ar):
    le_ar = MLB.inverse_transform(mlb_ar)
    other_lb = LE.transform(['other'])
    kw_le = [np.array(list(kw)) if kw!=tuple() else np.array(list(other_lb)) for kw in le_ar]
    return [list(LE.inverse_transform(kw_ar)) for kw_ar in kw_le]
#end
```

### Building the Classifier

A set of function are defined below to define and use a OneVsRest classifier, implemented by feeding a Random Forest classifier into the OneVsRest class from scikit-learn. This is equivalent to fitting a set of indepenent classifiers for each keywords, extracting the corresponding probabilities and consolidating everything into a vector of probabilities, representing the probability that each given keyword is associated to the given document.

```python
def set_max_kw(prob_ar):
    one_kw_ar = prob_ar.copy()
    one_kw_ar[np.arange(len(prob_ar)), prob_ar.argmax(1)] = 0.99
    return one_kw_ar
#end

def get_pred_lb(prob_ar, threshold):
    max_kw_ar = set_max_kw(prob_ar)
    pred_lb = np.zeros_like(max_kw_ar)
    pred_lb[max_kw_ar>=threshold] = 1
    return pred_lb
#end

def fit_clf(tfidf_smx, target_mlb):
    CLF = OneVsRestClassifier(RandomForestClassifier())
    CLF.fit(tfidf_smx, target_mlb)
    return CLF
#end

def predict_clf(CLF, tfidf_tst_smx, LE, MLB, thresh):
    tst_preds_prob = CLF.predict_proba(tfidf_tst_smx)
    tst_pred_lb_ar = get_pred_lb(tst_preds_prob, thresh)
    tst_pred_kw_ls = ml_invert_binarize(LE, MLB, tst_pred_lb_ar)
    return tst_pred_kw_ls
#end

def fit_predict_clf(tfidf_trn_smx, tfidf_tst_smx, target_mlb, LE, MLB, thresh):
    CLF = OneVsRestClassifier(RandomForestClassifier())
    CLF.fit(tfidf_trn_smx, target_mlb)
    tst_preds_prob = CLF.predict_proba(tfidf_tst_smx)
    tst_pred_lb_ar = get_pred_lb(tst_preds_prob, thresh)
    tst_pred_kw_ls = ml_invert_binarize(LE, MLB, tst_pred_lb_ar)
    return tst_pred_kw_ls, CLF
#end
```

## Results

The following steps follow the implementation described above. Separate results are provided for using just the title or the body of the document, as provided from the Ask Ubuntu dataset. As of now, the implementations are independent, but a simple next step could be to combine the two, possibly by learning the respective weight from data.

Notice also that the classifier outputs a set of probabilities for each document. A simple analysis (not shown here) showed that the optimal threshold for assigning a keyword to a given document is 0.2, so that is this value that is used in this document.

```python
source_path = './source_dataset/'
path = './'
s = 50000 #this value is necessary to downsize the training set for computational limitations
df = read_input()

train_df, test_df = get_train_test(df)
vocabulary_body = build_vocabulary(train_df, field='Body')
vocabulary_ttle = build_vocabulary(train_df, field='Title')


body_tfifd_trn_smx, body_tfifd_tst_smx = create_tfidf_smx(train_df['Body'].tail(s), test_df['Body'], vocabulary_body)
ttle_tfifd_trn_smx, ttle_tfifd_tst_smx = create_tfidf_smx(train_df['Title'].tail(s), test_df['Title'], vocabulary_ttle)
target_mlb, LE, MLB = build_le_mlb_trg(train_df.tail(s))
```

### Using the Title

```python
threshold = 0.2
tst_tt_pred_kw, CLF = fit_predict_clf(ttle_tfifd_trn_smx, ttle_tfifd_tst_smx, target_mlb, LE, MLB, threshold)
```
### Using the Body

```python
threshold = 0.2
tst_bd_pred_kw, CLF = fit_predict_clf(body_tfifd_trn_smx, body_tfifd_tst_smx, target_mlb, LE, MLB, threshold)
```

## Discussion

The approach above attains a Mean-F-Score of 0.32 and 0.30 using, respectively, just the title or just the body of the question. The result is encouraging, especially because it is obtained with a small training set, without considering title and body simultaneously, and with a very simple preprocessing of the dataset, and offers a good benchmark for further development and/or refinements.

## Appendix

### Imported Libraries

```python
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
```

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-101907146-1', 'auto');
  ga('send', 'pageview');

</script>