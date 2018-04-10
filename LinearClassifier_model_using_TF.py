
## Import dependencies

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.python.data import Dataset
from datetime import datetime

## Set some default values
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

## Import Data
path = 'data/talkingdata-adtracking-fraud-detection/'
train_df = pd.read_csv(path + 'train_sample.csv') 

train_df = train_df.reindex(
    np.random.permutation(train_df.index))

## Feature Engineering

# 1. Select features:
def preprocess_features(train_df):
  """Prepares input features from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the features to be used for the model, including
    synthetic features.
  """
  selected_features = train_df[
    ["ip",
     "app",
     "device",
     "os",
     "channel",
     "click_time"
     ]]
  processed_features = selected_features.copy()

  return processed_features

def preprocess_targets(train_df):
  """Prepares target features (i.e., labels) from California housing data set.

  Args:
    california_housing_dataframe: A Pandas DataFrame expected to contain data
      from the California housing data set.
  Returns:
    A DataFrame that contains the target feature.
  """
  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["is_attributed"] = train_df["is_attributed"]
  return output_targets

 # Choose the first 70000 (out of 100000) examples for training.
training_examples = preprocess_features(train_df.head(70000))
training_targets = preprocess_targets(train_df.head(70000))

# Choose the last 30000 (out of 30000) examples for validation.
validation_examples = preprocess_features(train_df.tail(30000))
validation_targets = preprocess_targets(train_df.tail(30000))

# Double-check that we've done the right thing.
print "Training examples summary:"
display.display(training_examples.describe())
print "Validation examples summary:"
display.display(validation_examples.describe())

print "Training targets summary:"
display.display(training_targets.describe())
print "Validation targets summary:"
display.display(validation_targets.describe())

## Feature Scaling, Feature crosses, etc..

def construct_feature_columns(input_features_num=None, input_features_cat=None,
                              feature_type=['numerical','cat_with_identy','cat_with_hash_bucket',
					    'cat_with_vocab_file','cat_with_vocab_list']):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features_num: The names of the numerical input features to use.
    input_features_cat: The names of the categorical input features to use.
    feature_type(list): The type of the feature categorical_with_identity/ categorical_with_hasBucket/
                                          categorical_with_vocabulary_file/ categorical_with_vocabulary_list/ numerical 
  Returns:
    A set of feature columns
  """ 
  if 'numerical' in feature_type:
    feature_cols = set([tf.feature_column.numeric_column(my_feature)
                          for my_feature in input_features_num])
  elif 'cat_with_identy' in feature_type:
    feature_cols.update(set([tf.feature_column.categorical_column_with_identity(my_feature)
                          for my_feature in input_features_cat]))
  elif 'cat_with_hash_bucket' in feature_type:
    feature_cols.update(set([tf.feature_column.categorical_column_with_hash_bucket(my_feature)
                          for my_feature in input_features_cat]))
  elif 'cat_with_vocab_file' in feature_type:
    feature_cols.update(set([tf.feature_column.categorical_column_with_vocabulary_file(my_feature)
                          for my_feature in input_features_cat]))
  elif 'cat_with_vocab_list' in feature_type:
    feature_cols.update(set([tf.feature_column.categorical_column_with_vocabulary_list(my_feature)
                          for my_feature in input_features_cat]))
  return feature_cols

def get_quantile_based_boundaries(feature_values, num_buckets):
  boundaries = np.arange(1.0, num_buckets) / num_buckets
  quantiles = feature_values.quantile(boundaries)
  return [quantiles[q] for q in quantiles.keys()]

def construct_feature_columns():
  """Construct the TensorFlow Feature Columns.

  Returns:
    A set of feature columns
  """ 
  households = tf.feature_column.numeric_column("households")
  longitude = tf.feature_column.numeric_column("longitude")
  latitude = tf.feature_column.numeric_column("latitude")
  housing_median_age = tf.feature_column.numeric_column("housing_median_age")
  median_income = tf.feature_column.numeric_column("median_income")
  rooms_per_person = tf.feature_column.numeric_column("rooms_per_person")
  
  # Divide households into 7 buckets.
  bucketized_households = tf.feature_column.bucketized_column(
    households, boundaries=get_quantile_based_boundaries(
      training_examples["households"], 7))

  # Divide longitude into 10 buckets.
  bucketized_longitude = tf.feature_column.bucketized_column(
    longitude, boundaries=get_quantile_based_boundaries(
      training_examples["longitude"], 10))
  
    # Divide latitude into 10 buckets
  bucketized_latitude = tf.feature_column.bucketized_column(
    latitude, boundaries=get_quantile_based_boundaries(
      training_examples["latitude"], 10))
  
  # Divide housing_median_age into  buckets
  bucketized_housing_median_age = tf.feature_column.bucketized_column(
    housing_median_age, boundaries=get_quantile_based_boundaries(
      training_examples["housing_median_age"], 5))
  
    # Divide median_income into  buckets
  bucketized_median_income = tf.feature_column.bucketized_column(
    median_income, boundaries=get_quantile_based_boundaries(
      training_examples["median_income"], 10))

  # Divide rooms_per_person into  buckets
  bucketized_rooms_per_person =tf.feature_column.bucketized_column(
    rooms_per_person, boundaries=get_quantile_based_boundaries(
      training_examples["rooms_per_person"], 10))
  
  # YOUR CODE HERE: Make a feature column for the long_x_lat feature cross
  long_x_lat = tf.feature_column.crossed_column(
      set([bucketized_longitude, bucketized_latitude]), hash_bucket_size=1000)
  
  feature_columns = set([
    bucketized_longitude,
    bucketized_latitude,
    bucketized_housing_median_age,
    bucketized_households,
    bucketized_median_income,
    bucketized_rooms_per_person,
    long_x_lat])
  
  return feature_columns

## Feature Normalization

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

def normalize_linear_scale(examples_dataframe):
  """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
  #
  # Your code here: normalize the inputs.
  return examples_dataframe.apply(linear_scale, axis=0)

# normalized_dataframe = normalize_linear_scale(preprocess_features(california_housing_dataframe))
# normalized_training_examples = normalized_dataframe.head(70000)
# normalized_validation_examples = normalized_dataframe.tail(30000)

def log_normalize(series):
  return series.apply(lambda x:math.log(abs(x+1.0)))

def clip(series, clip_to_min, clip_to_max):
  return series.apply(lambda x:(min(max(x, clip_to_min), clip_to_max)))

def z_score_normalize(series):
  mean = series.mean()
  std_dv = series.std()
  return series.apply(lambda x:(x - mean) / std_dv)

def binary_threshold(series, threshold):
  return series.apply(lambda x:(1 if x > threshold else 0))

def normalize(examples_dataframe, norm_type, clip_to_min=None, clip_to_max=None, threshold=None):
  """
  norm_type = 'log', 'clipping', 'z_score_norm', or 'binary_threshold'
    clip_to_min & clip_to_max only used in case of norm_type='clipping'
    threshold only used in case of norm_type='binary_threshold'
    
  Note: this function could be used to do different kind of normalization on different features
  by calling it multiple time for every group of features while specifying norm_type
  Returns a version of the input `DataFrame` that has all its features normalized."""
  #
  # YOUR CODE HERE: Normalize the inputs.
  if norm_type == 'log':
    return examples_dataframe.apply(log_normalize, axis=0)
  elif norm_type == 'clipping':
    return examples_dataframe.apply(clip, args=(clip_to_min, clip_to_max), axis=0)
  elif norm_type == 'z_score_norm':
    return examples_dataframe.apply(z_score_normalize, axis=0)
  elif norm_type == 'binary_threshold':
    return examples_dataframe.apply(binary_threshold, args=(threshold), axis=0)

# normalized_dataframe = normalize(preprocess_features(california_housing_dataframe), norm_type='log')
# normalized_dataframe = normalize(preprocess_features(california_housing_dataframe), clip_to_min=0, clip_to_max=1, norm_type='clipping')
# normalized_dataframe = normalize(preprocess_features(california_housing_dataframe), norm_type='z_score_norm')
normalized_dataframe = normalize(preprocess_features(california_housing_dataframe), threshold=[0.7], norm_type='binary_threshold')
normalized_training_examples = normalized_dataframe.head(12000)
normalized_validation_examples = normalized_dataframe.tail(5000)

def normalize_date(series):
	"""A function used to normalize click_time 
	   column data using Minmax scaler/ Standardscaler
	"""
	values = series.values
	values = values.reshape(len(values), 1)
	scaler = Standardscaler()
	scaler = scaler.fit(datetime.strptime(values[:10]))
	normalized = scaler.transform(datetime.strptime(values[:10]))
	return normalized
	# pass

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model of one feature.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.
      
  Returns:
    A `LinearClassifier` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a linear classifier object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_classifier = tf.estimator.LinearClassifier(feature_columns = construct_feature_columns(training_examples),
                                                    optimizer=my_optimizer)# YOUR CODE HERE: Construct the linear classifier.
  
  # Create input functions
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["median_house_value_is_high"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["median_house_value_is_high"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["median_house_value_is_high"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
  
  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print "Training model..."
  print "LogLoss (on training data):"
  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.    
    training_predictions = linear_classifier.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'] for item in training_probabilities])
    
    validation_predictions = linear_classifier.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'] for item in validation_probabilities])
    
    training_log_loss = metrics.log_loss(training_targets, training_predictions)
    validation_log_loss = metrics.log_loss(validation_targets, validation_predictions)
    # Occasionally print the current loss.
    print "  period %02d : %0.2f" % (period, training_log_loss)
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
  print "Model training finished."
  
  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()

  return linear_classifier


linear_classifier = train_linear_classifier_model(
						  learning_rate=0.000005,
						  steps=500,
						  batch_size=20,
						  training_examples=training_examples,
						  training_targets=training_targets,
						  validation_examples=validation_examples,
						  validation_targets=validation_targets)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print "AUC on the validation set: %0.2f" % evaluation_metrics['auc']
print "Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy']


validation_predictions = linear_classifier.predict(input_fn=predict_validation_input_fn)
# Get just the probabilities for the positive class
validation_predictions = np.array([item['predictions'][1] for item in validation_predictions])

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
    validation_targets, validation_predictions)
plt.plot(false_positive_rate, true_positive_rate, label="our model")
plt.plot([0, 1], [0, 1], label="random classifier")
_ = plt.legend(loc=2)
