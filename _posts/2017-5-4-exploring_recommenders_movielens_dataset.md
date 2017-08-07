---
layout: post
title: Exploring Recommender Systems
tags: machine-learning recommenders CNN Keras
---

Recommender systems are ubiquitous nowadays and they exploit patterns in people's preferences and tastes to provide personalized recommendations to users. Collaborative Filtering Recommenders (CFR) are possibly the most common and powerful engines and are widely used in a variety of domains. 

In this post, I explore several implementations of CFR, to give a taste of their capabilities and performances. To this aim, I use the movielens dataset, a freely available collection go user-based movie ratings that can be downloaded <a href="https://grouplens.org/datasets/movielens/" >here</a>. As the purpose of this post is primarily educational, I use the small dataset, which contais ~100,000 ratings and ~1,300 tags for ~9,000 movies and ~700 unique users.

As the movielens dataset is widely known and most of the techniques here implemented have a long history of application, this is only one of the many guides to recommender engines out there, and does not want to be the ultimate one by any means. On the contrary, the work I present in this post is inspired by many sources out there, such as the excellent <a href="http://www.fast.ai"> fast.ai MOOC</a>, which I encourage the reader to consult as well.

This document is organized as follows:

- Data Loading
- Exploratory Data Analysis
- Preliminary Manipulations
    - Train-Test Splitting
    - Performance Metric Definition
- Benchmarking
- Recommender Engines Implementation
    - Memory-Based Collaborative Filtering
        - Cosine Similarity
    - Model-Based Collaborative Filtering
        - Singular Value Decomposition
        - Matrices Multiplication
        - Matrices Multiplication with Bias
        - Neural Networks
- Summary
- Appendix
    - Imported Libraries
    - EDA graphs and tables

## Data Loading

I use pandas to load the datasets into DataFrames, then I produce a dictionary to build a unique sequencial indexing of users and movies and use it to add two unique index columns to the DataFrame.


```python
movies_df = pd.read_csv('./ml-latest-small/movies.csv')
ratings_df = pd.read_csv('./ml-latest-small/ratings.csv')
```

```python
def id2idx_dcs(id_ls):
    id2index_dc = {str(old_id): idx for idx, old_id in enumerate(id_ls)}
    index2id_dc = {str(idx): old_id for idx, old_id in zip(id2index_dc.values(), id2index_dc.keys())}
    return id2index_dc, index2id_dc
#end
```


```python
user_id2idx_dc, user_idx2id_dc = id2idx_dcs(list(ratings_df['userId'].unique()))
movie_id2idx_dc, movie_idx2id_dc = id2idx_dcs(list(ratings_df['movieId'].unique()))
ratings_df['userIDX'] = ratings_df['userId'].apply(lambda Id: user_id2idx_dc[str(Id)])
ratings_df['movieIDX'] = ratings_df['movieId'].apply(lambda Id: movie_id2idx_dc[str(Id)])
```

## Exploratory Data Analysis

Printing the first few rows of the DataFrames is a good way to learn about their structure and verify that they have been properly loaded.
It is also a good measure to make a few basic analyses and plot some basic graphs to get a feel of what kind of data is available. In order to avoid distracting from the focus of this post, however, most of the Exploratory Data Analysis (EDA), with both codes and graphs, is moved to the Appendix.


```python
ratings_df.head(3)
```

<div>
  <table class="dataframe">
    <thead>
      <tr>
        <th>userId</th>
        <th>movieId</th>
        <th>rating</th>
        <th>timestamp</th>
        <th>userIDX</th>
        <th>movieIDX</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>31</td>
        <td>2.5</td>
        <td>1260759144</td>
        <td>0</td>
        <td>0</td>
      </tr>
      <tr>
        <td>1</td>
        <td>1029</td>
        <td>3.0</td>
        <td>1260759179</td>
        <td>0</td>
        <td>1</td>
      </tr>
      <tr>
        <td>1</td>
        <td>1061</td>
        <td>3.0</td>
        <td>1260759182</td>
        <td>0</td>
        <td>2</td>
      </tr>
    </tbody>
  </table>
</div>

```python
movies_df.head(3)
```

<div>
  <table class="dataframe">
    <thead>
      <b>
        <tr>
          <th>movieId</th>
          <th>title</th>
          <th>genres</th>
        </tr>
      </b>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>Toy Story (1995)</td>
        <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      </tr>
      <tr>
        <td>2</td>
        <td>Jumanji (1995)</td>
        <td>Adventure|Children|Fantasy</td>
      </tr>
      <tr>
        <td>3</td>
        <td>Grumpier Old Men (1995)</td>
        <td>Comedy|Romance</td>
      </tr>
    </tbody>
  </table>
</div>


```python
n_users = len(ratings_df['userIDX'].unique())
n_movies = len(ratings_df['movieIDX'].unique())
print 'Number of Users: ' + str(n_users)
print 'Number of Movies: ' + str(n_movies)
print 'Matrix sparcity: ' + str(100.0 - (100.0 * len(ratings_df) / (n_users * n_movies))) + ' %'
```

```python
Number of Users: 671
Number of Movies: 9066
Matrix sparcity: 98.3560858391 %
```

## Preliminary Manipulations

### Train-Test Splitting

For all recommenders developed here, a 80-20 train-test split is used. In order to keep things consistent, the random split seed is fixed. 


```python
def train_test_split_df(input_df, ratio=0.8, msk=None, seed=None):
    if msk:
        train_df = input_df[msk]
        test_df = input_df[~msk]        
    else:
        np.random.seed(seed)
        msk = np.random.rand(len(input_df)) < ratio
        train_df = input_df[msk]
        test_df = input_df[~msk]
    #end
    return train_df, test_df
#end
```


```python
train_df, test_df = train_test_split_df(ratings_df, seed=7)
```

### Performance Metric Definition

Root Mean Squared Error between predicted ratings and observed ratings is used to assess the performance of each recommender. 


```python
def RMSE(pred_ar, truth_ar, matrix=True):
    if matrix:
        pred_ar = pred_ar[truth_ar.nonzero()].flatten()
        truth_ar = truth_ar[truth_ar.nonzero()].flatten()
    #end
    return sqrt(mean_squared_error(pred_ar, truth_ar))
#end
```

## Benchmarking

Whenever a new model or a new implementation of an old one is developed, it is good measure to find reasonable benchmarks against which the performance of the new implementation can be assessed. When working on a cutting-edge research topic, such benckmarks are usually the literature state of the art on the given field. However, basic benchmarks serve the purpose just as well since, in most cases, we want to have an idea of how much we are improving the simplest fast and dirty solution. To this aim, one can ask oneself: what is the simple estimate of the rating that a certain user is going to give to a certain movie? A good answer could be: the average rating that other users have given to the movie. Here, I use this principle to compute three benchmarks: 1) Each user will give a new movie the average of the ratings he or she gave so far; 2) Each user will give a new movie the average of the ratings other users have given to the movie; 3) Each given user will give a new movie the average of the average rating across movies for the given user and the average ratings for the new movie provided by other users. 


```python
def build_user2movie_matrix(df, n_users=None, n_movies=None):
    if not n_users:
        n_users = len(df['userIDX'].unique())
        n_movies = len(df['movieIDX'].unique())
    #end
    data_mx = np.zeros((n_users, n_movies))
    for index, row in df.iterrows():
        data_mx[int(row['userIDX'])][int(row['movieIDX'])] = row['rating']
    #end
    return data_mx
#end
```


```python
def benchmarks(train_df, test_df, n_users, n_movies, performance_dc=None, verbose=True):
    if not performance_dc:
        performance_dc = dict()
    #end
    train_mx = build_user2movie_matrix(train_df, n_users=n_users, n_movies=n_movies)
    mean_usr_rating_mx = train_mx.sum(axis=1) / np.count_nonzero(train_mx, axis=1)
    mean_mov_rating_mx = train_mx.sum(axis=0) / np.count_nonzero(train_mx, axis=0)
    truth_test_mx = np.zeros((n_users, n_movies))
    for index, row in test_df.iterrows():
        truth_test_mx[int(row['userIDX'])][int(row['movieIDX'])] = row['rating']
    #end
    # Benchmark 1: assigning mean user rating across movies to all new movies
    pred_test_mx = np.zeros((n_users, n_movies))
    pred_test_mx = pred_test_mx +  mean_usr_rating_mx[:, np.newaxis]
    pred_test_mx[np.where(np.isnan(pred_test_mx))] = np.nanmean(pred_test_mx)
    performance_dc['User Avg Benchmark'] = RMSE(pred_test_mx, truth_test_mx)
    if verbose:
        print 'User average benchmark RMSE: ' + str(RMSE(pred_test_mx, truth_test_mx))
    #end
    # Benchmark 2: assigning mean movie rating across users to all users
    pred_test_mx = np.zeros((n_users, n_movies))
    pred_test_mx = (pred_test_mx.T +  mean_mov_rating_mx[:, np.newaxis]).T
    pred_test_mx[np.where(np.isnan(pred_test_mx))] = np.nanmean(pred_test_mx)
    performance_dc['Movie Avg Benchmark'] = RMSE(pred_test_mx, truth_test_mx)
    if verbose:
        print 'Movie average benchmark RMSE: ' + str(RMSE(pred_test_mx, truth_test_mx))
    #end
    # Benchmark 3: averaging the mean user rating and the mean movie rating
    pred_test_mx = np.zeros((n_users, n_movies))
    pred_test_mx = 0.5 * (pred_test_mx +  mean_usr_rating_mx[:, np.newaxis] + \
                         (pred_test_mx.T +  mean_mov_rating_mx[:, np.newaxis]).T)
    pred_test_mx[np.where(np.isnan(pred_test_mx))] = np.nanmean(pred_test_mx)
    performance_dc['Averaged Movie-User Avg Benchmark'] = RMSE(pred_test_mx, truth_test_mx)
    if verbose:
        print 'Averaged User and Movie averages benchmark RMSE: ' + str(RMSE(pred_test_mx, truth_test_mx))
    #end
    return performance_dc
#end    
```


```python
performance_dc = benchmarks(train_df, test_df, n_users, n_movies)
```

```python
User average benchmark RMSE: 0.965366436781
Movie average benchmark RMSE: 0.999052599878
Averaged User and Movie averages benchmark RMSE: 0.929886343076
```

Based on the RMSE of the benchmarks above, a movie recommendation based on a simple procedure such as assigning the average ratings to unrated movies provides already a reasonable result. One may like to do better, though, so let us see what other routes we can explore.
 
## Recommender Engines Implementation

In the following paragraphs, I present the implementation of several recommender engines, as mentioned previously. In order to make the presentation consistent, I perform some initial manipulation of the dataset, splitting it into train and test and defining a metrics of performance for the validation of each approach.

### Memory-Based Collaborative Filtering

This type of recommenders are based on the assumption that past user behavior is predictive of future user behavior. Since user taste follows patterns, the identification of such patterns is essential to generate reliable predictions. Memory-Based Collaborative Filters exploit similarities between either a given user and other users (User-Based Collaborative Filters), or between items the given user likes and items that he or she may like as a result (Item-Based Collaborative Filters).

#### Cosine Similarity

In order to build the recommender system using the approach(es) above, one has to define a suitable metric to assess the similarity between users or items. Such metric is typically the cosine distance, calculated as the dot product of the users vectors or items vectors.

Both implementations (User-Based and Item-Based) require the construction of a user-item (in our case user-movie) matrix, that is, a matrix with as many rows as there are users and as many columns as there are movies. The cells of this matrix are the ratings that each individual user has given to each individual movie. Obviously, such matrix is in general very sparse, as most user rate, or even watch, only a fraction of the movies available.

Here are two routines to build the similarity matrix and use it to predict unknown ratings. The functions below can be used for both User-Based and Item-Based recommenders.

Notice that naive similarity computations do not take into account the fact that a given user can be inherently generous in rating movies, or a given movie can be inherently good (or bad). In order to cirvumvent this bias-like issue, I make predictions on average-adjusted matrices, that is, I first subtract the mean (computed excluing missing values), then add it back.


```python
def get_similarities(rating_data_mx, s_type='user'): 
    if s_type=='user':
        similarity_mx = 1 - pw_dist(rating_data_mx, metric='cosine')
    elif s_type=='item':
        similarity_mx = 1 - pw_dist(rating_data_mx.T, metric='cosine')
    #end
    return similarity_mx
#end

def predict(rating_mx, similarity_mx, s_type='user'):
    if s_type=='user':
        mean_user_rating_ar = rating_mx.sum(axis=1)/np.count_nonzero(rating_mx, axis=1)
        delta_ratings_mx = (rating_mx - mean_user_rating_ar[:, np.newaxis])
        delta_ratings_mx[rating_mx==0.0] = 0.0
        pred_ar = mean_user_rating_ar[:, np.newaxis] + similarity_mx.dot(delta_ratings_mx) / \
                  np.array([np.abs(similarity_mx).sum(axis=1)]).T
    elif s_type=='item':
        mean_item_rating_ar = rating_mx.sum(axis=0)/np.count_nonzero(rating_mx, axis=0)
        delta_ratings_mx = (rating_mx - mean_item_rating_ar[np.newaxis, :])
        delta_ratings_mx[rating_mx==0.0] = 0.0
        pred_ar = mean_item_rating_ar[np.newaxis, :] + delta_ratings_mx.dot(similarity_mx) / \
                  np.array([np.abs(similarity_mx).sum(axis=1)])
    #end
    return pred_ar
#end
```

Here I use the above functions to create the recommendations and assess the model(s) performance.


```python
train_rt_mx = build_user2movie_matrix(train_df, n_users=n_users, n_movies=n_movies)
truth_test_mx = build_user2movie_matrix(test_df, n_users=n_users, n_movies=n_movies)

usr_sim_mx = get_similarities(train_rt_mx, s_type='user')
mov_sim_mx = get_similarities(train_rt_mx, s_type='item')

movie_pred_mx = predict(truth_test_mx, mov_sim_mx, s_type='item')
user_pred_mx = predict(truth_test_mx, usr_sim_mx, s_type='user')
```


```python
print 'User-based Cosine Similarity RMSE: ' + str(RMSE(user_pred_mx, truth_test_mx))
print 'Item-based Cosine Similarity RMSE: ' + str(RMSE(movie_pred_mx, truth_test_mx))
performance_dc['User-based Cosine Similarity'] = RMSE(user_pred_mx, truth_test_mx)
performance_dc['Movie-based Cosine Similarity'] = RMSE(movie_pred_mx, truth_test_mx)
```

```python
User-based Cosine Similarity RMSE: 0.920526234083
Item-based Cosine Similarity RMSE: 0.815992426671
```

### Model-Based Collaborative Filtering

This category encompasses recommenders that use a model-based framework to build the recommender engine. While there are several of these, I here explore a few "Latent Factors" models plus a neural network model. Latent Factors models are based on the assumption that user rates a movie based on a set of features and that such features. These features are called "Latent Factors" because they are "hidden" or "implicit" in the ratings and can be "learned" from the ratings themselves. While there is no explicit way to tell what these features are, it can be assumed that they represent movies characteristics like the amount of action, the quality of the screenplay, the presence of famous actors, the presence of special effects and so on. The rating would ultimately be some sort of product between these factors and the weight that each users gives to each one of them, that is, how receptive a user is to the given latent factor.

While given a sufficient number of latent factors, it is possible to precisely fit the observations, generally fewer latent factors are selected in order to avoid overfitting. Therefore, Latent Factors Recommenders usually rely on defining a priory the number of features and apply some sort of minimization technique to learn these features from the known ratings. 

#### Singular Value Decomposition

Singular Value Decomposition (SVD) is a common technique of matrix decomposition often used, among other things, for dimensionality reduction. I will not provide a thorough description of the SVD implementation, one can read about it in textbooks or online everywhere, but the gist is to try to represent the user-movie rating matrix as the product of 3 matrices. Such matrices will represent the User latent factors, the Movie latent factors and the "relevance" of the given latent factor for the overall rating of a movie. Much like in dimensionality reduction, only the most relevant latent factors are be selected to build an SVD based recommender system.

A huge drawback of this implementation, is that the rating matrix is usually very sparse and SVD will try to decompose it by considering the missing values as zero. There are techniques to overcome this limitation, some of which are described <a href="https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf" >here</a>. As done previously with the cosine similarity approach, I apply SVD on the delta ratings matrix, that is, on the matrix defined as the difference between a given rating and the average rating of the given user.


```python
def singular_value_decompose(data_mx, n_sv=20):
    mean_user_rating_ar = data_mx.sum(axis=1) / np.count_nonzero(data_mx, axis=1)
    delta_ratings_mx = (data_mx - mean_user_rating_ar[:, np.newaxis])
    delta_ratings_mx[data_mx==0.0] = 0.0
    u, s, vt = svds(delta_ratings_mx, k=n_sv)
    s_diag_mx = np.diag(s)
    return u, s_diag_mx, vt
#end

def svd_predict(u, s_diag_mx, vt, data_mx):
    mean_user_rating_ar = data_mx.sum(axis=1) / np.count_nonzero(data_mx, axis=1)
    pred_mx = np.dot(np.dot(u, s_diag_mx), vt) + mean_user_rating_ar[:, np.newaxis]
    return pred_mx
#end

u, s_diag_mx, vt = singular_value_decompose(train_rt_mx, n_sv=50)
svd_pred_mx = svd_predict(u, s_diag_mx, vt, train_rt_mx)
print 'SVD MSE: ' + str(RMSE(svd_pred_mx, truth_test_mx))
performance_dc['SVD'] = RMSE(svd_pred_mx, truth_test_mx)
```

```python
SVD MSE: 0.945549294143
```

#### Matrices Multiplication

This Latent Factor Model is very similar to SVD, as they both share the idea that a rating can be decomposed as the product of latent factors.

Here, however, I make use of embeddings to exclude the missing values from the problem and provide a framework that only focuses on the known ratings. Embeddings are nothing more than lookup functions that, given a certain user and movie, will retrieve the respective latent factors vectors. Embeddings are learned from the ratings through minimization of a loss function, in this case the Mean Squared Error (MSE).

The implementation presented here is courtesy of Jeremy Howard at <a href="http://www.fast.ai"> fast.ai</a>, with only minor modifications, and makes use of the keras libraries and functional API.


```python
def embedding_input(emb_name, n_items, n_fact=20, l2regularizer=1e-4):
    inp = Input(shape=(1,), dtype='int64', name=emb_name)
    return inp, Embedding(n_items, n_fact, input_length=1, embeddings_regularizer=l2(l2regularizer))(inp)
#end

def build_dp_recommender(u_in, m_in, u_emb, m_emb):
    x = dot([u_emb, m_emb], axes=(2, 2))
    x = Flatten()(x)
    dp_model = Model([u_in, m_in], x)
    dp_model.compile(Adam(0.001), loss='mse')
    return dp_model
#end

usr_inp, usr_emb = embedding_input('user_inp', n_users, n_fact=50, l2regularizer=1e-4)
mov_inp, mov_emb = embedding_input('movie_inp', n_movies, n_fact=50, l2regularizer=1e-4)

dp_model = build_dp_recommender(usr_inp, mov_inp, usr_emb, mov_emb)

dp_model.fit([train_df['userIDX'], train_df['movieIDX']], train_df['rating'], batch_size=64, epochs=2,
             validation_data=([test_df['userIDX'], test_df['movieIDX']], test_df['rating']))

dp_model.optimizer.lr=0.01
dp_model.fit([train_df['userIDX'], train_df['movieIDX']], train_df['rating'], batch_size=64, epochs=3,
             validation_data=([test_df['userIDX'], test_df['movieIDX']], test_df['rating']))

dp_model.optimizer.lr=0.001
dp_model.fit([train_df['userIDX'], train_df['movieIDX']], train_df['rating'], batch_size=64, epochs=8,
             validation_data=([test_df['userIDX'], test_df['movieIDX']], test_df['rating']))

dp_preds_ar = np.squeeze(dp_model.predict([test_df['userIDX'], test_df['movieIDX']]))
print 'Dot Product MSE: ' + str(RMSE(dp_preds_ar, test_df['rating'].values, matrix=False)) 
performance_dc['Dot Product'] = RMSE(dp_preds_ar, test_df['rating'].values, matrix=False)
```

```python
Dot Product MSE: 1.21001928331
```

#### Matrices Multiplication with Bias Term

The matriced multiplication model through embeddings above did not produce very good results. This is likely due to the fact that the bias terms discussed before, that is, the inherent generosity of users and inherent quality of movies, are not factored in the model.

In the implementation below, a bias embedding is added to both users and movies as a simple additive term. All embeddings, both the latent factors and the biases, are then simultaneously learned through gradient descent with MSE loss function and Adam optimizer.


```python
def create_bias(inp, n_items):
    x = Embedding(n_items, 1, input_length=1)(inp)
    return Flatten()(x)
#end

def build_dp_bias_recommender(u_in, m_in, u_emb, m_emb, u_bias, m_bias):
    x = dot([u_emb, m_emb], axes=(2,2))
    x = Flatten()(x)
    x = add([x, u_bias])
    x = add([x, m_bias])
    bias_model = Model([u_in, m_in], x)
    bias_model.compile(Adam(0.001), loss='mse')
    return bias_model
#end

usr_inp, usr_emb = embedding_input('user_in', n_users, n_fact=50, l2regularizer=1e-4)
mov_inp, mov_emb = embedding_input('movie_in', n_movies, n_fact=50, l2regularizer=1e-4)

usr_bias = create_bias(usr_inp, n_users)
mov_bias = create_bias(mov_inp, n_movies)

bias_model = build_dp_bias_recommender(usr_inp, mov_inp, usr_emb, mov_emb, usr_bias, mov_bias)

bias_model.fit([train_df['userIDX'], train_df['movieIDX']], train_df['rating'], batch_size=64, epochs=2,
               validation_data=([test_df['userIDX'], test_df['movieIDX']], test_df['rating']))

bias_model.optimizer.lr=0.01
bias_model.fit([train_df['userIDX'], train_df['movieIDX']], train_df['rating'], batch_size=64, epochs=6,
               validation_data=([test_df['userIDX'], test_df['movieIDX']], test_df['rating']))

bias_model.optimizer.lr=0.001
bias_model.fit([train_df['userIDX'], train_df['movieIDX']], train_df['rating'], batch_size=64, epochs=15,
               validation_data=([test_df['userIDX'], test_df['movieIDX']], test_df['rating']))

bias_preds_ar = np.squeeze(bias_model.predict([test_df['userIDX'], test_df['movieIDX']]))

print 'Dot Product Model with Bias MSE: ' + str(RMSE(bias_preds_ar, test_df['rating'].values, matrix=False))
performance_dc['Dot Product with Bias'] = RMSE(bias_preds_ar, test_df['rating'].values, matrix=False)
```

```python
Dot Product Model with Bias MSE: 0.9085571397
```

#### Neural Network Model

In this last section, a standard neural network is built on the concatenated user and movie embeddings. This has the flexibility of neural network architecture and, hopefully, will be able to implicitly reproduce the effect of latent factors and biases through the non linearities of the neural network, without the need of an ad-hoc architecture.


```python
def build_nn_recommender(u_in, m_in, u_emb, m_emb, d_out=0.25, dens_f=64):
    x = concatenate([u_emb, m_emb])
    x = Flatten()(x)
    x = Dropout(d_out)(x)
    x = Dense(dens_f, activation='relu')(x)
    x = Dropout(3*d_out)(x)
    x = Dense(1)(x)
    nn_model = Model([u_in, m_in], x)
    nn_model.compile(Adam(0.001), loss='mse')
    return nn_model
#end

usr_inp, usr_emb = embedding_input('user_in', n_users, n_fact=50, l2regularizer=1e-4)
mov_inp, mov_emb = embedding_input('movie_in', n_movies, n_fact=50, l2regularizer=1e-4)

nn_model = build_nn_recommender(usr_inp, mov_inp, usr_emb, mov_emb)

nn_model.fit([train_df['userIDX'], train_df['movieIDX']], train_df['rating'], batch_size=64, epochs=6,
             validation_data=([test_df['userIDX'], test_df['movieIDX']], test_df['rating']))

nn_preds_ar = np.squeeze(nn_model.predict([test_df['userIDX'], test_df['movieIDX']]))

print 'Neural Network MSE: ' + str(RMSE(nn_preds_ar, test_df['rating'].values, matrix=False)) 
performance_dc['Neural Network'] = RMSE(nn_preds_ar, test_df['rating'].values, matrix=False)
```

```python
Neural Network MSE: 0.890825204386
```

## Summary

Let us now compare the set of models built.


```python
perf_df = pd.DataFrame(performance_dc.items(), columns=['Recommender', 'RMSE'])
perf_df.sort_values(by='RMSE', ascending=True).reset_index(drop=True)
```

<div>
  <table class="dataframe">
    <thead>
      <tr>
        <th>Recommender</th>
        <th>RMSE</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Movie-based Cosine Similarity</td>
        <td>0.815992</td>
      </tr>
      <tr>
        <td>Neural Network</td>
        <td>0.890825</td>
      </tr>
      <tr>
        <td>Dot Product with Bias</td>
        <td>0.908557</td>
      </tr>
      <tr>
        <td>User-based Cosine Similarity</td>
        <td>0.920526</td>
      </tr>
      <tr>
        <td>Averaged Movie-User Avg Benchmark</td>
        <td>0.929886</td>
      </tr>
      <tr>
        <td>SVD</td>
        <td>0.945549</td>
      </tr>
      <tr>
        <td>User Avg Benchmark</td>
        <td>0.965366</td>
      </tr>
      <tr>
        <td>Movie Avg Benchmark</td>
        <td>0.999053</td>
      </tr>
      <tr>
        <td>Dot Product</td>
        <td>1.210019</td>
      </tr>
    </tbody>
  </table>
</div>


Interestingly enough, most models are not performing much better than the best benchmark and the best performer, by far, is the Item-based Collaborative Filter. This means that, at least at this stage of model prototyping and with this dataset, the best recommender engine would be based on the concept that, if you like a certain movie, you may also like another movie that is similar - according to fellow users - to it.

Obviously, there is room to improve the current implementations, asses their robustness, and to explore other architectures.

## Appendix

### Imported Libraries


```python
import numpy as np
import pandas as pd
import pickle
import keras
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances as pw_dist
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.sparse.linalg import svds
from keras.models import Model
from keras.layers import Input, Embedding
from keras.layers.core import Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.merge import dot, add, concatenate
```

### EDA tables and plots


```python
ratings_df.describe()
```


<div>
<table class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>userIDX</th>
      <th>movieIDX</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>100004.000000</td>
      <td>100004.000000</td>
      <td>100004.000000</td>
      <td>1.000040e+05</td>
      <td>100004.000000</td>
      <td>100004.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>347.011310</td>
      <td>12548.664363</td>
      <td>3.543608</td>
      <td>1.129639e+09</td>
      <td>346.011310</td>
      <td>1660.778349</td>
    </tr>
    <tr>
      <th>std</th>
      <td>195.163838</td>
      <td>26369.198969</td>
      <td>1.058064</td>
      <td>1.916858e+08</td>
      <td>195.163838</td>
      <td>1893.955817</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>7.896520e+08</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>182.000000</td>
      <td>1028.000000</td>
      <td>3.000000</td>
      <td>9.658478e+08</td>
      <td>181.000000</td>
      <td>327.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>367.000000</td>
      <td>2406.500000</td>
      <td>4.000000</td>
      <td>1.110422e+09</td>
      <td>366.000000</td>
      <td>873.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>520.000000</td>
      <td>5418.000000</td>
      <td>4.000000</td>
      <td>1.296192e+09</td>
      <td>519.000000</td>
      <td>2344.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>671.000000</td>
      <td>163949.000000</td>
      <td>5.000000</td>
      <td>1.476641e+09</td>
      <td>670.000000</td>
      <td>9065.000000</td>
    </tr>
  </tbody>
</table>
</div>


```python
def unpack_movies_df(movies_df, ratings_df):
    full_unpacked_df = movies_df.merge(ratings_df, on='movieId')
    full_unpacked_df['genres'] = full_unpacked_df['genres'].apply(lambda g: g.split('|'))
    full_unpacked_df = pd.concat([pd.DataFrame(dict(zip(full_unpacked_df.columns, \
                                                        full_unpacked_df.ix[i]))) for i in range(len(full_unpacked_df))] \
                                 ).reset_index(drop=True)
    return full_unpacked_df
#end

full_unpacked_df = unpack_movies_df(movies_df, ratings_df)
full_unpacked_df.head(3)
```

<div>
<table class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>genres</th>
      <th>movieIDX</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>title</th>
      <th>userIDX</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adventure</td>
      <td>417</td>
      <td>1</td>
      <td>3.0</td>
      <td>851866703</td>
      <td>Toy Story (1995)</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Animation</td>
      <td>417</td>
      <td>1</td>
      <td>3.0</td>
      <td>851866703</td>
      <td>Toy Story (1995)</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Children</td>
      <td>417</td>
      <td>1</td>
      <td>3.0</td>
      <td>851866703</td>
      <td>Toy Story (1995)</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>


```python
def plot_hist(ratings_df):
    fig = plt.figure(figsize=(16, 6))
    ax = sns.countplot(x='rating', data=ratings_df, color='#4c4c4c')
    ax.set_xlabel('Rating', fontsize=14)
    ax.set_ylabel('Number of reviews', fontsize=14)
    return
#end

def plot_breakdown_hist(full_df):
    fig = plt.figure(figsize=(16, 10))
    ax = sns.countplot(x='rating', data=full_df, hue='genres')
    ax.set_xlabel('Rating', fontsize=14)
    ax.set_ylabel('Number of reviews', fontsize=14)
    return    
#end

def plot_ratings_dist(ratings_df, groupby='movie'):
    if groupby=='movie':
        group_id = 'movieIDX'
        aggregate_id = 'userIDX'
        x_label = 'Reviews per movie'
        xlim_ls = [0, 100]
    elif groupby=='user':
        group_id = 'userIDX'
        aggregate_id = 'movieIDX'
        x_label = 'Reviews per user'
        xlim_ls = [20, 120]
    #end
    grouped_df = ratings_df.groupby([group_id]).agg({aggregate_id: 'count'}).sort_values(by=aggregate_id, 
                                                                                         ascending=False)
    #end
    fig = plt.figure(figsize=(16, 6))
    for fig_id in range(2):
        ax = fig.add_subplot(1, 2, fig_id + 1)
        ax = sns.distplot(grouped_df[aggregate_id], bins=grouped_df[aggregate_id].max(), color='k')
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel('pdf', fontsize=14)
    #end
    ax.set_xlim(xlim_ls)
    ax.set_ylim([0, 0.05])
    return
#end

def plot_genre_ratings(full_df):
    fig = plt.figure(figsize=(16, 6))
    ax = sns.barplot(x='genres', y='rating', data=full_df, ci=None)
    ax.set_xlabel('Genre', fontsize=14)
    ax.set_ylabel('Rating', fontsize=14)
    plt.xticks(rotation=45)
    return
#end
```


```python
%matplotlib inline
sns.set(style='whitegrid')

plot_genre_ratings(full_unpacked_df)
plot_hist(ratings_df)
plot_breakdown_hist(full_unpacked_df)
plot_ratings_dist(ratings_df, groupby='movie')
plot_ratings_dist(ratings_df, groupby='user')
```

{% include image.html img="images/recommender_systems_post/movielens_recommenders_80_0.png" title="movielens rating by genre" width="900" %}


{% include image.html img="images/recommender_systems_post/movielens_recommenders_81_0.png" title="movielens rating histogram" width="900" %}


{% include image.html img="images/recommender_systems_post/movielens_recommenders_82_0.png" title="movielens rating distribution breakdown by genre" width="900" %}


{% include image.html img="images/recommender_systems_post/movielens_recommenders_83_0.png" title="movielens reviews per movie" width="900" %}


{% include image.html img="images/recommender_systems_post/movielens_recommenders_84_0.png" title="movielens reviews per user" width="900" %}

<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-101907146-1', 'auto');
  ga('send', 'pageview');

</script>