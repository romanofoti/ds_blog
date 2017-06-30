---
layout: post
title: Walkthrough of an exploratory analysis for classification problems
---

In this post I outline how to perform an exploratory analysis for a binary classification problem.

I am going to analyze a dataset provided as a .csv file. The dataset contains a set of column of possibile predictors plus one response column (binary). No information about where the dataset comes from or what each predictor is (nor the response) is given. So, what does one do when asked to analyze a dataset - and possibly provide interpretations and actionable insights - knowing nothing except the dataset itself? Here is a possible approach:

 - Configure your working environment and load the input
 - Perform some basic exploratory data analysis (EDA)
 - Manipulate the data and perform some basic feature engineering
 - Prototyping and compare classification models
 - Choose a model and further refine the analysis
 - Identify and discuss possible actionable insights

Each one of the the above steps is developed and discussed in the various sections of this notebook. An appendix with additional information is also provided.

## 1. Environment configuration and input loading

The code in this section is used to configure the workspace, preference and to load the input. All the modules used in the data challenge are defined and imported here.


```python
#%%******************************************************************************
# Importing packages
#******************************************************************************
import numpy as np  #library for matrix-array analysis
import pandas as pd  #library for advanced data analysis
import matplotlib.pyplot as plt #library to plot graphs
import pickle #library to handle input/output
import seaborn as sns #seaborn wrapper plotting library
from sklearn.preprocessing import StandardScaler #import the module to perform standardization
from sklearn.decomposition import PCA #import the module to perform Principal Component Analysis
from sklearn.cross_validation import train_test_split #import package to create the train and test dataset
from sklearn.linear_model import LogisticRegression #import package to perform Logistic Regression
from sklearn.ensemble import RandomForestClassifier #import package to perform Random Forest
from sklearn.ensemble import GradientBoostingClassifier #import package to perform Gradient Boosting
from sklearn.neighbors import KNeighborsClassifier #import package to perform k-NN classifier
from sklearn.metrics import precision_score, recall_score, f1_score #import metrics score to validate algorithms
import sys
from sklearn.metrics import confusion_matrix as CM #import the confusion matrix package to evaluate classification performance
from sklearn import cross_validation #import cross-validation module
from sklearn.learning_curve import learning_curve #import learning curve module
from sklearn.metrics import precision_recall_curve #import precision-recall curve
import bisect #import module to provide support for maintaining a list in sorted order without having to sort the list after each insertion
from scipy.stats import mstats #import module to evaluate some statistical objects
#******************************************************************************

#-------------------------------------------
# Jupyter Notebook settings
#-------------------------------------------
%matplotlib inline
sns.set(style="darkgrid")
#-------------------------------------------

#-------------------------------------------
# Define the paths
#-------------------------------------------
input_path = './'
onput_path = input_path
#-------------------------------------------

#-------------------------------------------
# Loading the input in a pandas dataframe
#-------------------------------------------
input_df = pd.read_csv(input_path + 'dataset.csv')
#*********************************************

```

## 2. Exploratory Data Analysis

In this section, I explore the dataset provided. Several considerations can be made:

- The column of responses is binary
- There are 41 predictors, none of which have informative headers
- There are over 125k columns
- The percentage of positive resopnses is about 7%, suggesting a unbalanced dataset
- The predictors vary largely in scale and distribution

In performing the exploratory analysis, I took the following steps:

- Evaluating cleanliness of the dataset and handling missing data
- Exploring relationships among variables

Results and insights of this sections are reported in the subsections below.

### 2.1 Input description and missing data handling


```python
input_df.describe()
```

<div style="max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>response</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>var_7</th>
      <th>var_8</th>
      <th>var_9</th>
      <th>...</th>
      <th>var_32</th>
      <th>var_33</th>
      <th>var_34</th>
      <th>var_35</th>
      <th>var_36</th>
      <th>var_37</th>
      <th>var_38</th>
      <th>var_39</th>
      <th>var_40</th>
      <th>var_41</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>127945.000000</td>
      <td>120144.000000</td>
      <td>1.201440e+05</td>
      <td>120144.000000</td>
      <td>120144.000000</td>
      <td>120144.000000</td>
      <td>120144.000000</td>
      <td>120144.000000</td>
      <td>120144.000000</td>
      <td>120144.000000</td>
      <td>...</td>
      <td>1.201440e+05</td>
      <td>120144.000000</td>
      <td>120144.000000</td>
      <td>120144.000000</td>
      <td>120144.000000</td>
      <td>1.201440e+05</td>
      <td>120144.000000</td>
      <td>122653.000000</td>
      <td>101368.000000</td>
      <td>100784.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.069327</td>
      <td>2.618483</td>
      <td>2.225702e+04</td>
      <td>1.303444</td>
      <td>1.034142</td>
      <td>39.796777</td>
      <td>0.437242</td>
      <td>0.043664</td>
      <td>0.646932</td>
      <td>980.538912</td>
      <td>...</td>
      <td>8.244423e+04</td>
      <td>0.801621</td>
      <td>1.080986</td>
      <td>0.141505</td>
      <td>0.307423</td>
      <td>3.246416e+04</td>
      <td>0.274595</td>
      <td>5671.021622</td>
      <td>744274.423447</td>
      <td>2000.758900</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.254010</td>
      <td>8.222190</td>
      <td>1.192116e+06</td>
      <td>4.891512</td>
      <td>3.577857</td>
      <td>10173.365626</td>
      <td>5.474776</td>
      <td>10.932956</td>
      <td>10.140853</td>
      <td>21812.114958</td>
      <td>...</td>
      <td>3.177394e+06</td>
      <td>9.449891</td>
      <td>25.594473</td>
      <td>2.708312</td>
      <td>1.403094</td>
      <td>1.418447e+06</td>
      <td>1.592004</td>
      <td>1692.196258</td>
      <td>772732.764947</td>
      <td>10.145973</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>36.000000</td>
      <td>1984.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>4451.000000</td>
      <td>260000.000000</td>
      <td>1992.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>5419.000000</td>
      <td>472000.000000</td>
      <td>2005.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000e+02</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>7225.000000</td>
      <td>896000.000000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>397.000000</td>
      <td>1.257448e+08</td>
      <td>218.000000</td>
      <td>144.000000</td>
      <td>3099209.000000</td>
      <td>117.000000</td>
      <td>3171.000000</td>
      <td>623.000000</td>
      <td>2543002.000000</td>
      <td>...</td>
      <td>3.003248e+08</td>
      <td>504.000000</td>
      <td>2888.000000</td>
      <td>401.000000</td>
      <td>54.000000</td>
      <td>1.232632e+08</td>
      <td>76.000000</td>
      <td>9251.000000</td>
      <td>20500000.000000</td>
      <td>2013.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 42 columns</p>
</div>


Estimating how many columns contain missing data.


```python
inds = pd.isnull(input_df).any(1).nonzero()[0] #create an array with the index of the rows where missing data are present
print 1.0*len(inds)/len(input_df)
```

```python
0.229411075071
```

Exploring the characteristics of the dataset for those rows where missing data is present.


```python
input_df.iloc[inds].describe()
```

<div style="max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>response</th>
      <th>var_1</th>
      <th>var_2</th>
      <th>var_3</th>
      <th>var_4</th>
      <th>var_5</th>
      <th>var_6</th>
      <th>var_7</th>
      <th>var_8</th>
      <th>var_9</th>
      <th>...</th>
      <th>var_32</th>
      <th>var_33</th>
      <th>var_34</th>
      <th>var_35</th>
      <th>var_36</th>
      <th>var_37</th>
      <th>var_38</th>
      <th>naics</th>
      <th>revenue</th>
      <th>year_established</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>29352.000000</td>
      <td>21551.000000</td>
      <td>2.155100e+04</td>
      <td>21551.000000</td>
      <td>21551.000000</td>
      <td>21551</td>
      <td>21551.000000</td>
      <td>21551</td>
      <td>21551.000000</td>
      <td>21551.000000</td>
      <td>...</td>
      <td>2.155100e+04</td>
      <td>21551.000000</td>
      <td>21551.000000</td>
      <td>21551.000000</td>
      <td>21551.000000</td>
      <td>21551.000000</td>
      <td>21551.000000</td>
      <td>24060.000000</td>
      <td>2775.000000</td>
      <td>2191.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.131303</td>
      <td>3.662521</td>
      <td>4.429236e+04</td>
      <td>1.904598</td>
      <td>1.500951</td>
      <td>0</td>
      <td>0.548884</td>
      <td>0</td>
      <td>0.883764</td>
      <td>1382.479328</td>
      <td>...</td>
      <td>1.876692e+05</td>
      <td>1.270103</td>
      <td>1.639599</td>
      <td>0.202821</td>
      <td>0.447589</td>
      <td>40678.163426</td>
      <td>0.405503</td>
      <td>5631.332585</td>
      <td>788402.601802</td>
      <td>2004.561387</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.337737</td>
      <td>10.737826</td>
      <td>1.948302e+06</td>
      <td>6.277614</td>
      <td>4.617769</td>
      <td>0</td>
      <td>5.951514</td>
      <td>0</td>
      <td>12.177012</td>
      <td>26723.702892</td>
      <td>...</td>
      <td>5.585162e+06</td>
      <td>12.658793</td>
      <td>26.146182</td>
      <td>2.536659</td>
      <td>1.752328</td>
      <td>1480748.144161</td>
      <td>2.031644</td>
      <td>1796.243732</td>
      <td>940130.453493</td>
      <td>9.312571</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>6460.000000</td>
      <td>1984.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4441.000000</td>
      <td>261000.000000</td>
      <td>2001.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000e+00</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5241.000000</td>
      <td>473000.000000</td>
      <td>2009.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>2.000000e+02</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7225.000000</td>
      <td>932500.000000</td>
      <td>2012.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>312.000000</td>
      <td>1.257448e+08</td>
      <td>167.000000</td>
      <td>107.000000</td>
      <td>0</td>
      <td>117.000000</td>
      <td>0</td>
      <td>577.000000</td>
      <td>2190962.000000</td>
      <td>...</td>
      <td>3.003248e+08</td>
      <td>475.000000</td>
      <td>1549.000000</td>
      <td>179.000000</td>
      <td>47.000000</td>
      <td>93893900.000000</td>
      <td>76.000000</td>
      <td>9251.000000</td>
      <td>16300000.000000</td>
      <td>2013.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 42 columns</p>
</div>


<b>Note:</b> Eliminating the ~23% of rows containing missing data would leave us with ~100k columns, probably sufficient not to lose information. However, looking at the distribution of responses in the subsection of the dataset with missing data, one can see that positive responses now account for ~13% of the total. This suggest that it is not wise to ignore rows with missing data altogether. I choose to keep those rows and fill the missing data in the given cell with the mean of the respective column. 


```python
input_filled_df = input_df.fillna(input_df.mean())
```

### 2.2. Exploring relationships among variables

#### 2.2.1. Visualizing pairplots and distribution of predictors and response

Notice that a full pairplot is too big for visualization. An acceptable preliminary analysis can be done by looking at sections of pairwise scatter plots, as done below.


```python
fig1 = plt.figure(figsize=(16,16));
sns.pairplot(input_filled_df.iloc[:,np.hstack(([0],range(1,11)))],diag_kind='kde',hue='response',palette='Set1');
fig2 = plt.figure(figsize=(16,16));
sns.pairplot(input_filled_df.iloc[:,np.hstack(([0],range(11,21)))],diag_kind='kde',hue='response',palette='Set1');
fig3 = plt.figure(figsize=(16,16));
sns.pairplot(input_filled_df.iloc[:,np.hstack(([0],range(21,31)))],diag_kind='kde',hue='response',palette='Set1');
fig4 = plt.figure(figsize=(16,16));
sns.pairplot(input_filled_df.iloc[:,np.hstack(([0],range(31,42)))],diag_kind='kde',hue='response',palette='Set1');
```
{% include image.html img="images/eca_post_imgs/c1.png" title="corr1" width="900" %}

{% include image.html img="images/eca_post_imgs/c2.png" title="corr2" width="900" %}

{% include image.html img="images/eca_post_imgs/c3.png" title="corr3" width="900" %}

{% include image.html img="images/eca_post_imgs/c4.png" title="corr4" width="900" %}

<b> Insights: </b>
- The number of negative responses, in blue, usually masks the distribution of positive responses, making it difficult to identify patterns
- Many variables appear to be strongly correlated (e.g. var_26 and var 27 may actually be the same predictor twice)
- The distributions of predictors are variable in shape
- Some predictors display different distributions in when the response is positive than when the response is negative

#### 2.2.1. Further exploring correlations:

```python
heatmap_corr(input_filled_df) #custom built function to produce a heatmap using seaborn
```

{% include image.html img="images/eca_post_imgs/heat_map.png" title="heatmap" width="900" %}



```python
input_filled_df.plot(kind='scatter', x='var_3',y='var_4')
```

{% include image.html img="images/eca_post_imgs/sc1.png" title="scatter1" width="600" %}


```python
input_filled_df.plot(kind='scatter', x='var_26',y='var_27')
```

{% include image.html img="images/eca_post_imgs/sc2.png" title="scatter2" width="600" %}


## 3. Data manipulation and feature engineering

### 3.1. Dimensionality reduction

Given the high correlation displayed among some of the predictors, I make an attempt to reduce the dimensionality via principal component analysis. The results shows that over 98% of the variance in the data is explained by the first 27 components.
While this approach is at the expense of interpretability of the predictors, the reduction in complexity is significant.


```python
#-------------------------------------------
# Separate the predictors (signals) from the response
#-------------------------------------------
signals = input_filled_df[[c for c in input_filled_df.columns if c != 'response']]
responses = input_filled_df['response']
#-------------------------------------------

#-------------------------------------------
# Performing standardization and PCA
#-------------------------------------------
var_threshold = 0.98 # minimum percentage of variance we want to be described by the resulting transformed components
pca_obj = PCA(n_components=var_threshold) # Create PCA object
signals_Transformed = pca_obj.fit_transform(StandardScaler().fit_transform(signals)) # Transform the initial features
columns = ['comp_' + str(n) for n in range(1,signals_Transformed.shape[1]+1)] #create a list of columns
transf_signals_df = pd.DataFrame(signals_Transformed, columns=columns) # Create a data frame from the PCA'd data
transf_input_df = transf_signals_df.copy()
transf_input_df['response'] = responses #create a full dataframe (including the response) out of the transformed features
#-------------------------------------------
print signals_Transformed.shape
```

```python
(127945, 27)
```

Notice that now the heatmap shows no correlation among features.


```python
heatmap_corr(transf_signals_df)
```

{% include image.html img="images/eca_post_imgs/heat_map2.png" title="heatmap_no_corr" width="900" %}


### 3.2. Dealing with class imbalance

Model performance can be affected by imbalance in the responses. Most models have out-of-the-box built-in functions (e.g. autoweight) to penalize wrong predicitions of the less represented class. I make use of that. However, I also resample to rebalance classes. In this exercise, I perform a one-time random undersampling of the dataset. The resulting dataset will be significantly smaller, but this will actually make the computations faster and more suitable for prototyping.


```python
#---------------------------
# Resample (undersampling to balance categories)
#---------------------------
n_positive = len(responses[responses==1])
resampled_df = transf_input_df[transf_input_df.response==1].copy()
df = transf_input_df[transf_input_df.response==0].sample(n_positive,replace = False)
resampled_df = resampled_df.append(df, ignore_index=True)

#---------------------------
# Resample (undersampling to allow speedy computation)
#---------------------------

resampled_signals_df = resampled_df[[c for c in resampled_df.columns if c != 'response']].values
resampled_response_sr = resampled_df['response'].values
#---------------------------
```

## 4. Model prototyping and comparison among models

A set of models are compared out of the box to identify the most promising ones for further development.

The chosen models are:

- Logistic Regression
- k-NN
- Random Forest
- Gradient Boosting

Performance is estimated through precision, recall and f-1 score, using 25 separate runs for each model. For each run, a test set is built with 20% of the data. A computationally more expensive model, SVM, is here neglected for the sake of speed.


```python
#******************************************************************************
# Models Comparison
#******************************************************************************
#---------------------------
# Building a list of classifier_dc to compare
#---------------------------
classifier_ls = ['Logistic_Regression', 'Random_Forest', 'Gradient_Boosting', 'K_NN']
classifier_dc = {
               'Logistic_Regression': LogisticRegression(class_weight='auto'),
               'Random_Forest': RandomForestClassifier(n_estimators=50, class_weight='auto'),
               'Gradient_Boosting': GradientBoostingClassifier(max_depth=5),
               'K_NN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
               }
n_trials = 25
test_size_rt = 0.2
#---------------------------

#---------------------------
# Calculating model performances for different models and several cv sets
#---------------------------
score_ls = []
for ic, (cl_name, Classifier) in enumerate(classifier_dc.items()):
	for trial in range(n_trials):
		train_signals, test_signals, train_labels, test_labels = train_test_split(resampled_signals_df, resampled_response_sr, test_size=test_size_rt)
		Classifier.fit(train_signals, train_labels)
		pred_labels = Classifier.predict(test_signals)
		precision = precision_score(test_labels, pred_labels, average='binary')
		score_ls.append([cl_name, precision, 'Precision'])
		recall = recall_score(test_labels, pred_labels, average='binary')
		score_ls.append([cl_name, recall,'Recall'])
		F1 = f1_score(test_labels, pred_labels, average='binary')
		score_ls.append([cl_name, F1,'f1 Score'])
		if (trial+1)%5==0:
			print (trial+1)*4, '%'
			sys.stdout.flush()
		#end
	#end
#end
```

Plotting the results (and saving the dataframe).


```python
#---------------------------
# Plotting boxplots of model performances
#---------------------------
metrics_df = pd.DataFrame(score_ls, columns=['Classifier','Score','Score_type']) #building a dataframe with various metrics
ax = sns.boxplot(x="Classifier", y="Score", hue="Score_type", data=metrics_df, linewidth=2.5) #producing the boxplot
metrics_df.to_pickle(onput_path + 'metrics_df.pkl') #saving results
```

{% include image.html img="images/eca_post_imgs/model_comparison.png" title="model_comparison" width="800" %}


<b>Insights:</b>

Tree-based algorithms appear to be the best performers. The metrics are not outstanding and could be improved by further feature engineering and tuning, though. Random Forest will be used as a model of choice for further validation, the choice based on compromise between speed and performance.

## 5. Validation of chosen model

The Random Forest classifier is further validated by means of:

- Confusion Matrix
- Cross-validated ROC curve
- Learnign Curves


```python
#******************************************************************************
# Validating Chosen Classifier
#******************************************************************************

#---------------------------
# Splitting into train and test sets
#---------------------------
test_size_rt = 0.2
train_signals, test_signals, train_labels, test_labels = train_test_split(resampled_signals_df, resampled_response_sr, test_size=test_size_rt)

#---------------------------
# Setting the classifier
#---------------------------
random_state = np.random.random_state(0) # Define a random state
Classifier = RandomForestClassifier(n_estimators=100, class_weight='auto', random_state=random_state)
```

### 5.1. Confusion Matrix


```python
#---------------------------
# Producing the Confusion Matrix
#---------------------------
Classifier.fit(train_signals, train_labels) #fitting the classifier
predicted_responses = Classifier.predict(test_signals) #applying predictions
conf_mat = CM(test_labels, predicted_responses,np.unique(train_labels)) #building the confusion matrix
labels = np.unique(train_labels.astype(int).astype(str)).tolist() #extracting the labels
sns.set_style('white') #setting the plotting style
plot_confusion_matrix(conf_mat, labels, Norm='True', Cmap=plt.cm.gray_r, Fig_counter=1, Title='Random Forest Confusion Matrix') #calls the confusion matrix routine with the test set and prediction set
precision = precision_score(test_labels, predicted_responses, average='binary')
recall = recall_score(test_labels, predicted_responses, average='binary')
print 'Precision = ', '{:.2f}'.format(precision)
print 'Recall = ', '{:.2}'.format(recall)
```

```python
Precision =  0.67
Recall =  0.62
```

{% include image.html img="images/eca_post_imgs/CM.png" title="Confusion matrix" width="500" %}


### 5.2. Cross-validated ROC curve


```python
#---------------------------
# Plotting Cross Validated ROC
#---------------------------
cross_valid_roc(resampled_signals_df,resampled_response_sr,Classifier, Folds = 5, title = '5-fold CV ROC curve') #calling the function to plot the cross-validated ROC
#---------------------------
```


{% include image.html img="images/eca_post_imgs/ROC5.png" title="ROC" width="800" %}


### 5.3. Learning Curves


```python
#---------------------------
# Plotting learning curve
#---------------------------
test_size_rt = 0.2
random_state = np.random.random_state(0) # Define a random state
Classifier = RandomForestClassifier(n_estimators=100, class_weight='auto', random_state=random_state, oob_score=True)
title = "Learning Curves for Random Forest Classifier"
cross_val = cross_validation.ShuffleSplit(resampled_signals_df.shape[0], n_iter=10, \
	                               test_size=test_size_rt, random_state=random_state) # Cross validation with 10 iterations and 20% test size
plot_learning_curves(Classifier, resampled_signals_df, resampled_response_sr, title = title, ylim=(0.0, 1.01), cv=cross_val)
#---------------------------
```

{% include image.html img="images/eca_post_imgs/learning_c.png" title="learning curves" width="600" %}


<b>Insights:</b>

None of the metrics above are outstanding at this point. The model appears to be overfitting and the ROC curve shows that the confidence on the predictions is not great.

A graph of feature importance is provided below. At this point, however, this is not particularly meaningful, as it would be necessary to transform the features back to the original predictors to understand which of them play the most important roles in the outcome.


```python
#---------------------------
# Create a bar plot of feature importance according to GINI index
#---------------------------
feat_ls = resampled_df.keys().tolist()
importance_ls = Classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in Classifier.estimators_], axis=0)
importance_ls, feat_ls, std = (list(t) for t in zip(*sorted(zip(importance_ls, feat_ls, std), reverse=True)))
plt.figure(figsize=(7,5))
plt.title("Feature importances")
plt.bar(range(len(feat_ls[0:6])), importance_ls[0:6], color='gray', yerr=std[0:6], ecolor='black', align="center")
plt.xticks(range(len(feat_ls[0:6])), feat_ls[0:6], rotation='vertical')
plt.xlim([-0.5, len(feat_ls[0:6])-0.5])
plt.savefig(onput_path + 'FeatureImportance.png', dpi=600, bbox_inches='tight')
```


{% include image.html img="images/eca_post_imgs/eca_13.png" title="gini importance" width="500" %}


## 6. Identification and investigation of actionable insights

In this section, I identify a possible business problem (that is, I assume that the response variable has a business meaning) and use the model results to inform a potential business decision.

<b> Business Problem: </b>

Let us assume that the response represents customer curn on a particular website/subscription. Obviously, something like a churn rate would be important to predict, so that some customer retention measures can be taken. For the sake of the argument, I will make the following assumptions:

- A value of 1 in the response variable corresponds to a customer that unsuscribes
- We want our model to be able to identify the customers that are more likely to unsuscribe based on a certain measurable metrics
- Each customer that is identified as "likely to unsubscribe" will trigger some "retention action" (for instance, a deal is offered to the customer)
- There is a cost for each "retention action", which implies that we want to minimize the number of actions taken, especially for customers that were wrongly identified as likely to unsuscribe

<b> Question: </b>

How do I set a classification threshold so as to optimize the combination of cost of retention actions and costs of losing the customer?

<b> Approach: </b>

I will look a the following factors:


<b> Precision: </b> What's the downside of erroneously identifying a customer as likely to unsuscribe (false positive)? A low precision will cost time, money and could potentially hurt the relationship with the customer.

<b> Recall: </b> What's the downside when you fail to take action for a customer that would otherwise unsuscribe (false negative)? A low recall will result in neglect customers that unsuscribe.

<b> Review Rate: </b> How many actions can be taken overall? This depends on the cost of treating an individual case (how much offering a one-time deal to a customer that is aboud to unsucribe will cost our business), as well as on the overall capacity.


## 6.1. Precision-Recall-Review curves

To address all these factors at once, a chart that shows precision, recall and review rate as a function of the classifier threshold is produced below. This allows to identify the tradeoffs for different thresholds and to make the optimal business decision.


```python
#******************************************************************************
# Actionable Insights: Thresholding (1/2)
#******************************************************************************
#---------------------------
# Define the classifier and get the predictions
#---------------------------
Classifier = RandomForestClassifier(n_estimators=100, class_weight='auto', random_state=random_state, oob_score=True)  
test_size_rt = 0.2
#---------------------------

#---------------------------
# Make a simple plot of the precision, Recalla and Review curves
#---------------------------
train_signals, test_signals, train_labels, test_labels = train_test_split(resampled_signals_df, resampled_response_sr, test_size=test_size_rt)  
Classifier.fit(train_signals, train_labels)
predictions = Classifier.predict_proba(test_signals)[:,1]  

precision_ls, recall_ls, thresholds_ls = precision_recall_curve(test_labels, predictions) #retieve the precision, recall and corresponding thresholds
thresholds_ls = np.append(thresholds_ls, 1) #add the last datapoint to the threshold list

Review_rate = [] #Initialize the queue rate as an empty list
for threshold in thresholds_ls:  
	review_rate_ls.append((predictions >= threshold).mean()) #estimate how many predictions would be made as positive for a given threshold
#end

plt.plot(thresholds_ls, precision_ls, color=sns.color_palette()[0]) #plot the precision curve as a fuction of the set threshold
plt.plot(thresholds_ls, recall_ls, color=sns.color_palette()[1]) #plot the Recall curve as a fuction of the set threshold
plt.plot(thresholds_ls, review_rate_ls, color=sns.color_palette()[2]) #plot the Review rate curve as a fuction of the set threshold
Legend = plt.legend(('precision', 'Recall', 'Review_rate'), frameon=True, loc='best') 
Legend.get_frame().set_edgecolor('k')  
plt.xlabel('Threshold')  
plt.ylabel('Proportion')  
#---------------------------
```

{% include image.html img="images/eca_post_imgs/prec_recall.png" title="precision-recall curves" width="800" %}



<b> Insights: </b>

Precision, Recall and Review Rate are expressed as a proportion of total number of cases for each threshold selected for the classifier. For exaple, chosing the default threshold of 0.5 would imply:

- A "retention action" is taken in about 50% of the cases
- Precision is ~60%, meaning that 60% of the customers that are predicted as likely to unsuscribe ultimately will
- Recall is ~50%, which implies that about 50% of the customers that will unsuscribe will not be considered for any "retention action"

The optimal choice of the threshold ultimately depends on the physical and financial constraints of the business.

For instance, if the business is limited to be able to give a one-time deal only to ~15% of the customers, a threshold of about 0.8 will need to be chosen, in which case:

- Precision is ~82%, meaning that 82% of the customers identified as potential unsuscribers would actually have unsuscribed outherwise
- Recall is ~22%, which implies that about 78% of the businesses that will unsuscribe will not be given any deal


## 6.2. precision-Recall-Review_rate curves: uncertainty analysis

The chart above only shows the performance of a single train/test split. Below, I present the distribution of precision, recall and review rate for an ensemble of random train/test splits, in order to get an idea of the range of possible performance outcomes. The solid lines represent the median precision, recall and review rates, while the shaded areas represent the respective 10% and 90% quantiles.


```python
#******************************************************************************
# Actionable Insights: Thresholding (2/2)
#******************************************************************************

#---------------------------
# Dealing with uncertainty in the model
#---------------------------
n_trials = 20 #define the number of random trials
plot_data = [] #define a list of dictionaries to store the curves to be plotted

for trial in range(n_trials): 
	train_signals, test_signals, train_labels, test_labels = train_test_split(resampled_signals_df, resampled_response_sr, test_size=test_size_rt)  
	Classifier.fit(train_signals, train_labels)
	predictions = Classifier.predict_proba(test_signals)[:,1]

	precision, Recall, Thresholds = precision_recall_curve(test_labels, predictions)  
	Thresholds = np.append(Thresholds, 1)

	Review_rate = []
	for threshold in Thresholds:
	    Review_rate.append((predictions >= threshold).mean())
	#end

	#-----------------------------
	# Append the curves as a dictionary entry of a list
	#-----------------------------
	plot_data.append({
	        'Thresholds': Thresholds,
	        'precision': precision,
	        'Recall': Recall,
	        'Review_rate': Review_rate
	})
	#-----------------------------
#end

uniform_thresholds = np.linspace(0, 1, 101) #define an array of thresholds

uniform_precision_plots = [] #initialize the list of lists to load the precision curve
uniform_recall_plots= [] #initialize the list of lists to load the recall curve
uniform_review_rate_plots= [] #initialize the of lists list to load the review rate curve

for p in plot_data:  
	Uniform_precision = [] #initialize the list loading the precision curve
	Uniform_recall = [] #initialize the list loading the recall curve
	Uniform_review_rate = [] #initialize the list loading the review rate curve
	for ut in uniform_thresholds:
		index = bisect.bisect_left(p['Thresholds'], ut) #retrieving the index corresponding to the given threshold value
		Uniform_precision.append(p['precision'][index]) #retieving the precision corresponding to the given threshold value 
		Uniform_recall.append(p['Recall'][index]) #retieving the recall corresponding to the given threshold value 
		Uniform_review_rate.append(p['Review_rate'][index]) #retieving the review rate corresponding to the given threshold value 
	#end

	uniform_precision_plots.append(Uniform_precision) #append the list of precision curve values
	uniform_recall_plots.append(Uniform_recall) #append the list of recall curve values
	uniform_review_rate_plots.append(Uniform_review_rate) #append the list of review rate curve values
#end

Quantiles = [0.1, 0.5, 0.9] #define the quantiles to plot curves and shading areas
lower_precision, median_precision, upper_precision = mstats.mquantiles(uniform_precision_plots, Quantiles, axis=0) #extract the precision quantiles for each threshold
lower_recall, median_recall, upper_recall = mstats.mquantiles(uniform_recall_plots, Quantiles, axis=0) #extract the recall quantiles for each threshold
lower_review_rate, median_review_rate, upper_review_rate = mstats.mquantiles(uniform_review_rate_plots, Quantiles, axis=0) #extract the review rate quantiles for each threshold

#-----------------------------
# Plot curves and fill between quantiles
#-----------------------------
plt.plot(uniform_thresholds, median_precision) #plots the median precision curve
plt.plot(uniform_thresholds, median_recall) #plots the median recall curve
plt.plot(uniform_thresholds, median_review_rate) #plots the median review rate curve

plt.fill_between(uniform_thresholds, upper_precision, lower_precision, alpha=0.5, linewidth=0, color=sns.color_palette()[0]) 
plt.fill_between(uniform_thresholds, upper_recall, lower_recall, alpha=0.5, linewidth=0, color=sns.color_palette()[1])  
plt.fill_between(uniform_thresholds, upper_review_rate, lower_review_rate, alpha=0.5, linewidth=0, color=sns.color_palette()[2])
#-----------------------------

Legend = plt.legend(('precision', 'Recall', 'Review_rate'), frameon=True, loc='best') 
Legend.get_frame().set_edgecolor('k')  
plt.xlabel('Threshold')  
plt.ylabel('Proportion') 
#-----------------------------

```

{% include image.html img="images/eca_post_imgs/prec_recall_uncer.png" title="precision-recall uncertainty curves" width="800" %}


<b> Insights: </b>

There is not a great variability across random splits. In this case, for the threshold 0.8 as above, it is expected:

- A 90% probability that the cases where an action is taken will be between 14% and 16%
- Precision is between 78% and 81% with 90% probability
- Recall is between 21% and 23% with 90% probability


<b> Concluding remarks </b>

- A more complete analysis of the optimal threshold would require knowledge of the costs associated with taking action, as well as losing the customer altogether
- A better model would have better precision and recall metrics across all thresholds, which would result in smaller review rates for each desired precision and recall level.

## 7. Appendix

### 7.1. Functions used in the model


```python
#******************************************************************************
# Define the functions
#******************************************************************************

def heatmap_corr(df):
	#-----------------------------------------------
	# Creates a heatmap of correlation from a dataframe
	#-----------------------------------------------
	import seaborn as sns #seaborn wrapper plotting library
	plt.figure() #initialize the figure
	corrmat = df.corr() # build the matrix of correlation from the dataframe using pandas.corr() function
	f, ax = plt.subplots(figsize=(12, 9)) 	# set up the matplotlib figure
	sns.heatmap(corrmat, vmax=1.0, vmin=-1.0, square=True) 	# draw the heatmap using seaborn
#end

def plot_confusion_matrix(CM, labels, Norm='True', Cmap=plt.cm.Blues, Fig_counter=1, Title='Confusion Matrix'):
#******************************************************************************
    #Plots the confusion matrix as a chessed graph with colors instead of numbers
    #INPUT: 1) Confusion Matrix
    #       2) Vector of labels
    #       3) Normalization of Confusion Matrix
    #       4) Colormap
    #       5) Title of plot
    #OUTPUT: Confusion matrix plot
#******************************************************************************
	if Norm == 'True':
	    CM = CM.astype('float')/CM.sum(axis=0)[np.newaxis,:] #Normalize the matrix along the TRUE label axis
	#end
	plt.figure(Fig_counter,figsize=(7,5))
	plt.imshow(CM, interpolation='nearest', cmap=Cmap) #create the graph and set the interpolation
	plt.title(title) #adding the title
	plt.colorbar() #additing the colorbar
	if Norm == 'True':
	   plt.clim(0,1) #Set the colorbar limits
	#end    
	tick_marks = np.arange(len(labels)) #defininig the tick marks
	plt.xticks(tick_marks, labels) #apply the labels to marks
	plt.yticks(tick_marks, labels) #apply the labels to marks
	plt.ylabel('True label') #adding the y-axis title
	plt.xlabel('Predicted label') #adding the x-axis title
#end

def cross_valid_roc(X, y, Classifier, Folds = 5, Title = ''):

	from scipy import interp
	from sklearn import svm, datasets
	from sklearn.metrics import roc_curve, auc
	from sklearn.cross_validation import StratifiedKFold

	# Run classifier with cross-validation and plot ROC curves
	cv = StratifiedKFold(y, n_folds=5)

	mean_tpr = 0.0
	mean_fpr = np.linspace(0, 1, 100)
	all_tpr = []

	for i, (train, test) in enumerate(cv):
	    probas_ = Classifier.fit(X[train], y[train]).predict_proba(X[test])
	    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
	    mean_tpr += interp(mean_fpr, fpr, tpr)
	    mean_tpr[0] = 0.0
	    roc_auc = auc(fpr, tpr)
	    plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))
	#end

	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

	mean_tpr /= len(cv)
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)
	plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title(title)
	plt.legend(loc="lower right")
#end

def plot_learning_curves(Estimator, X, y, title='', ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)):
#******************************************************************************
	"""
	Generate a plot of the test and traning learning curve.

	Parameters
	----------
	Estimator : object type that implements the "fit" and "predict" methods
	    An object of that type which is cloned for each validation.

	title : string
	    Title for the chart.

	X : array-like, shape (n_samples, n_features)
	    Training vector, where n_samples is the number of samples and
	    n_features is the number of features.

	y : array-like, shape (n_samples) or (n_samples, n_features), optional
	    Target relative to X for classification or regression;
	    None for unsupervised learning.

	ylim : tuple, shape (ymin, ymax), optional
	    Defines minimum and maximum yvalues plotted.

	cv : integer, cross-validation generator, optional
	    If an integer is passed, it is the number of folds (defaults to 3).
	    Specific cross-validation objects can be passed, see
	    sklearn.cross_validation module for the list of possible objects
	"""
#******************************************************************************
	plt.figure()
	plt.title(title)
	if ylim is not None:
	    plt.ylim(*ylim)
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	train_sizes, train_scores, test_scores = learning_curve(Estimator, X, y, cv=cv, train_sizes=train_sizes)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)
	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
	                 train_scores_mean + train_scores_std, alpha=0.1,
	                 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
	                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
	         label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
	         label="Cross-validation score")

	plt.legend(loc="best")
	return plt
#end

```

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-101907146-1', 'auto');
  ga('send', 'pageview');

</script>