## Supervised
- lasso.py     *: Predict with a Lasso Regression model*
- knn.py       *: Predict with a k-Nearest Neighbors model*
- bayes.py     *: Predict with a Bayesian model*
- gaussian.py  *: Predict with a Gaussian Process model*
- svm.py       *: Predict with a Support Vector Machine model*
- tree.py      *: Predict with a Decision Tree model*
- forest.py    *: Predict with a Random Forest model*
- xgboost.py   *: Predict with a XGBoost Tree model*
- keras.py     *: Predict with a Neural Network model*
- subsemble.py *: Predict with an Ensemble of models and partitions*
- blend.py     *: Predict with an Ensemble of models*
- pipe_lasso.py *: Predict with a Lasso Regression pipeline*
- pipe_nnet.py  *: Predict with a Neural Network pipeline*

## Model Tuning
- tune_knn.py *: Tunes a k-Nearest Neighbors model with a random grid search*
- tune_svm.py *: Tunes a Support Vector Machine model with a random grid search*
- tune_tree.py *: Tunes a Decision Tree model with a random grid search*
- tune_forest.py *: Tunes a Random Forest model with a random grid search*
- tune_xgboost.py *: Tunes a XGBoost Tree model with a random grid search*
- doe.R *: Selects an optimal subset of a grid search*

## Unsupervised
- kmeans.py *: Cluster with a k-Means model*
- hclust.py *: Cluster with a Hierarchical Clustering model*
- birch.py *: Cluster with a Birch model*
- mixture.py *: Cluster with a Gaussian Mixture model*
- mean.py *: Cluster with a Mean Shift model*
- pca.py *: Embed with a Principal Component Analysis model*
- isomap.py *: Embed with a Isomap model*
- lle.py *: Embed with a Locally Linear Embedding model*

## Preprocessing
- clean.py *: Fill in missing values, make all values numeric*
- outliers.py *: Remove outliers*
- features.py *: Generate features and select features*
- features2.py *: Generate features and select features*
- timeLag.py *: Add time-lagged features to features*

## Time Series
- lstm.py *: Forecast with a Long Short Term Memory Neural Network model*
- hmm.py *: Forecast (states) with a Hidden Markov Model*
- arima.py *: Rolling forecast with an Autoregressive Integrated Moving Average model* (Regression Only)
- exp.py *: Rolling forecast with a Simple Exponential Smoothing model* (Regression Only)
- holt.py *: Rolling forecast with a Holt-Winter's model* (Regression Only)
- rolling_bayes.py *: Rolling forecast with a Bayesian Ridge Regression model* (Regression Only)

## Natural Language Processing
- words.py *: Rank words on how well they predict text clusters (topics)*

## Graphics
- plots.py *: Plot with seaborn*
- plotting.py *: Plot with plot.ly*
