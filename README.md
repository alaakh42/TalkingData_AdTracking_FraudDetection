# TalkingData_AdTracking_FraudDetection
A kaggle competition to predict whether a user will download an app after clicking a mobile app ad
  You can download the data from the competion [page](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data)
I built an XGBoost model that is fitted in a numerical encoded data, The model was trained using ```55000000``` records of the train dataset.

It took time between 25:45 minutes (According to Data Size) as the algorithm implementation is very robust even with very large datatest. 

After some excessive hyperparameters tuning, I got AUC of 0.9638 ln the public Leaderboard

The Model Predictions (My 2 Top score Submissions):
- The submission file in this Google Drive [link](https://drive.google.com/drive/u/0/folders/0B70KdP1JxlcTVkVZNG9OOWxleTQ)

LightGBM model has too many hyperparameters and those needs carefull tuning, the best way to do that is using Grid Search Optimization 

According to LightGBM [documentation](https://lightgbm.readthedocs.io/en/latest/) hasTo increase the accuracy of a LightGBM model, to increase the prediction accuracy of the algorithm you can start by doing one of the following:

1. Use large ```max_bin``` (but it may be slower)
2. Use small ```learning_rate``` with large ```num_iterations```
3. Use large ```num_leaves``` (but it may cause over-fitting)
4. Use bigger training data
5. Try dart, You can choose the ```boosting_type``` of the algorithm between ```gbdt```, ```dart```, ```rf``` or ```goss``` 

To Do::
  - TalkingData company is providing a huge amount of data ```200 million``` record , if you have an enought powerful machine, you diffently would want to train using the whole dataset.
  - You can try tp implement a deep learning model for such a huge data
  - Try to do more feature engineering and see if results can get better
  - Try to do downsampling due to the inbalanced ratio of the fraud and non-fraud records 
