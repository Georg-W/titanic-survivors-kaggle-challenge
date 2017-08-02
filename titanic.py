# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:56:00 2017

@author: Georg
"""

import numpy as np
import pandas as pd
import tempfile
import tensorflow as tf

train_dataset = pd.read_csv("train.csv", na_values=[])
test_dataset = pd.read_csv("test.csv", na_values=[])

COLUMNS = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
LABEL_COLUMN = ["Survived"]


CATEGORICAL_COLUMNS = ["Pclass", "Name", "Sex",
                       "Ticket", "Cabin", "Embarked"]
CONTINUOUS_COLUMNS = ["PassengerId","SibSp", "Age", "Parch", "Fare"]

train_dataset["Pclass"].fillna(1, inplace=True)
train_dataset["Cabin"].fillna("", inplace=True)
train_dataset["Sex"].fillna("male", inplace=True)
train_dataset["Ticket"].fillna("", inplace=True)
train_dataset["Embarked"].fillna("", inplace=True)
train_dataset["Name"].fillna("", inplace=True)
train_dataset["SibSp"].fillna(0, inplace=True)
train_dataset["Age"].fillna(0, inplace=True)
train_dataset["Parch"].fillna(0, inplace=True)
train_dataset["Fare"].fillna(0, inplace=True)


test_dataset["Pclass"].fillna(1, inplace=True)
test_dataset["Cabin"].fillna("", inplace=True)
test_dataset["Sex"].fillna("male", inplace=True)
test_dataset["Ticket"].fillna("", inplace=True)
test_dataset["Embarked"].fillna("", inplace=True)
test_dataset["Name"].fillna("", inplace=True)
test_dataset["SibSp"].fillna(0, inplace=True)
test_dataset["Age"].fillna(0, inplace=True)
test_dataset["Parch"].fillna(0, inplace=True)
test_dataset["Fare"].fillna(0, inplace=True)
#for c in COLUMNS:
#    counter = 0
#    for data in train_dataset[c]:
#        if pd.isnull(data):
#            if c == "Name" or "Sex" or "Ticket" or "Cabin" or "Embarked":
#                train_dataset[c][counter] = "null"
#            else:
#                train_dataset[c][counter] = 0
#        counter = counter+1
       
#for c in COLUMNS:
##    counter = 0
#    for data in test_dataset[c]:
#        if pd.isnull(data):
#            if c == "Name" or "Sex" or "Ticket" or "Cabin" or "Embarked":
#                test_dataset[c][counter] = "null"
#            else:
#                test_dataset[c][counter] = 0
#        counter = counter+1


#p_class = tf.contrib.layers.sparse_column_with_keys(
#  column_name="Pclass", keys=[1, 2, 3])
# p_class = tf.contrib.layers.sparse_column_with_hash_bucket("Pclass", hash_bucket_size=1000)
name = tf.contrib.layers.sparse_column_with_hash_bucket("Name", hash_bucket_size=1000)
sex = tf.contrib.layers.sparse_column_with_keys(
  column_name="Sex", keys=["male", "female"])
ticket = tf.contrib.layers.sparse_column_with_hash_bucket("Ticket", hash_bucket_size=1000)
cabin = tf.contrib.layers.sparse_column_with_hash_bucket("Cabin", hash_bucket_size=1000)
embarked = tf.contrib.layers.sparse_column_with_keys(
  column_name="Embarked", keys=["C", "Q", "S"])


passenger_id = tf.contrib.layers.real_valued_column("PassengerId")
age = tf.contrib.layers.real_valued_column("Age")
sibsp = tf.contrib.layers.real_valued_column("SibSp")
parch = tf.contrib.layers.real_valued_column("Parch")
fare = tf.contrib.layers.real_valued_column("Fare")

#age_buckets = tf.contrib.layers.bucketized_column(age, boundaries=[16, 25, 40, 60])


def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  print("before data")
  print(continuous_cols.items())
  print(categorical_cols.items())
  print("after data")

  # Merges the two dictionaries into one.
  # + categorical_cols.items()
  #feature_cols = dict(categorical_cols.items())
  feature_cols = categorical_cols.copy()
  feature_cols.update(continuous_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label

def pred_input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in CATEGORICAL_COLUMNS}
  print("before data")
  print(continuous_cols.items())
  print(categorical_cols.items())
  print("after data")

  # Merges the two dictionaries into one.
  # + categorical_cols.items()
  # continuous_cols.items()
  #feature_cols = dict(categorical_cols.items()+continuous_cols.items())
  feature_cols = categorical_cols.copy()
  feature_cols.update(continuous_cols)
  
  # Converts the label column into a constant Tensor.
  #label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols

def train_input_fn():
  return input_fn(train_dataset)

def prediction_input_fn():
  return pred_input_fn(test_dataset)




model_dir = tempfile.mkdtemp()
m = tf.contrib.learn.LinearClassifier(feature_columns=[
  passenger_id, sibsp, parch, fare,
  age, name, sex, ticket, cabin, embarked],
      
      # p_class,
      
      
            #name, sex, ticket, cabin, embarked],
  #    p_class, name, sex, ticket, cabin, embarked, passenger_id, sibsp, parch, fare,
  #age],
  model_dir=model_dir)

m.fit(input_fn=train_input_fn, steps=20000)

pred = m.predict(input_fn=prediction_input_fn)
print(pred)
newpred = []
for p in pred:
    newpred.append(p)
    
print(newpred)
    
test_dataset['Survived'] = pd.Series(newpred, index = test_dataset.index)
test_dataset.to_csv("./pred.csv", encoding= 'utf-8', index = False, columns=["PassengerId", "Survived"])

#results = m.evaluate(input_fn=eval_input_fn, steps=1)
#for key in sorted(results):
 #   print("%s: %s" % (key, results[key]))

