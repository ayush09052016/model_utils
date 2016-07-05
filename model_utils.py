from __future__ import division
from dateutil.parser import *
from datetime import *
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score

# Function to derive age from a set of dates or w.r.t. the current date
def get_age_from_date(date):
        if date is None or type(date) == 'float':
            date = datetime.now()
        return (datetime.now()- parse(date)).days
        
# this is the main function that performs data preprocessing. this function makes calls to sub functions that
# that perform sub tasks
def get_pre_processed_dataset(df):
    # dropping place_id because there are too many place ids in the logs which doesn't really give any extra info. Instead,
    # having Merchant identifier as a feature may have valuable info and may serve as an approx. for the item category.
    # dropping job stage since its irrelevant for fraud prediction
    # dropping item description because none of the recorded fraud transactions have this field populated
    # excluding billing zip because we don't have billing info for any of the good txns. Only bad txns have biling info
    # which is sent to us by Stripe -- need to be investigated further
    df.drop(['job_id','merchant','place_id', 'item_description', 'job_stage','billing_zip',
             'dropoff_zip', 'market_id'], axis = 1, inplace = True)

    # replace the NaNs with 0s.
    df = replace_missing_values(df)
    # reduce the bad_type column to have binary values. (1 for fraudulent chargeback, 0 for everything else)
    df = make_bad_type_binary(df)
    # add a label column to the dataset
    df = create_label(df)

    # drop the screened, stopped and bad_type columns from the dataset
    df.drop(['screened', 'stopped', 'bad_type'], axis = 1, inplace = True)

    # get customer age from customer created date
    cx_age = df['cx_created_dt'].map(get_age_from_date)
    # get the age of the 1st job from the first job date
    first_job_age = df['first_job_dt'].map(get_age_from_date)
    # get the age of the last job from the last job date
    last_job_age = df['last_job'].map(get_age_from_date)

    # add the age columns to the dataset
    df['cx_age'] = cx_age
    df['first_job_age'] = first_job_age
    df['last_job_age'] = last_job_age

    #drop the date columns since they are of no use.
    df.drop(['cx_created_dt', 'first_job_dt', 'last_job'], axis = 1, inplace = True)

    # get dummy columns for the categorical features in the dataset
    cols = ['delivery_type','market', 'client_type']
    df = get_dummy_cols(df, cols)

    # remove the categorical columns once dummy variables have been derived from them
    df.drop(['delivery_type', 'market', 'client_type'], axis = 1, inplace = True)

    return df
    
def replace_missing_values(df):
    # making corrections for null or missing values
    df['first_job_dt'].replace(value = str(datetime.now()), to_replace = np.nan, inplace = True)
    df['last_job'].replace(value = str(datetime.now()), to_replace = np.nan, inplace = True)
    df['tip'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['estimated_total_purch_price'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['purchase_price'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['tip'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['total_jobs'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['rating_by_cust'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['rating_by_courier'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['screened'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['stopped'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['is_all_jobs_in_30'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['is_frequent'].replace(value = 0, to_replace = np.nan, inplace = True)
    df['been_frequent'].replace(value = 0, to_replace = np.nan, inplace = True)
    
    return df
    
def make_bad_type_binary(df):

    # mark all non-fraudulent transactions as '0'
    df['bad_type'].replace(value = 0, to_replace = [np.nan, 'unrecognized','general','product_not_received'
                           ,'credit_not_processed','product_unacceptable', 'duplicate','subscription_canceled'],
                         inplace = True)
    # mark all fraudulent transactions as '1'
    df['bad_type'].replace(value = 1, to_replace = 'fraudulent',
                          inplace = True)
    return df

#create a label column that marks the example as fraud (label = 1) if (bad_type == 1 OR stopped == 1). basically we
# want to mark a job as fraudulent if either we receive a chargebak or if the agent marked it as #FRAUD.

def create_label(df):
    cond = df['stopped'] == 1
    cond1 = df[cond |(df['bad_type'] == 1)]
    idx = cond1.index.values

    df['label'] = pd.Series()

    df.set_value(idx, 'label', 1)

    df['label'].replace(value = 0, to_replace = np.nan, inplace = True)
    
    return df

def get_sub_sample(df,good_sample_size ):
    # take a random sub sample of the good transactions
    cond = df['label']== 0
    df_good = df[cond]
    #sampler = np.random.permutation(3000)
    df_good_sample = df_good.take(np.random.permutation(len(df_good)))[:good_sample_size]
    # take 100% of the bad transactions
    cond1 = df['label'] == 1
    df_bad = df[cond1]
    df_final = pd.concat([df_good_sample, df_bad])
    return df_final
    
#run the bias variance test by plotting the learning curves
def get_learning_curves(X_train, X_cv, y_train, y_cv, start_with, step_size, classifier):
    l_error_train = []
    l_error_cv = []
    for m in range(start_with, len(X_train), step_size):
        X_m = X_train[:m]
        y_m = y_train[:m]
        classifier.fit(X_m,y_m)

        # using cross_validation to generate average scores for K folds
        #print 'avg training error = ', 1- cross_val_score(clf,X_cv, y_cv, cv = 10).mean()
        # for generating metrics manually for each fold
        predictions_train = clf.predict(X_m)
        predictions_cv = clf.predict(X_cv)

        err_train = 1- precision_score(y_m, predictions_train)
        err_cv = 1- precision_score(y_cv, predictions_cv)

        l_error_train.append(err_train)
        l_error_cv.append(err_cv)
        
    return l_error_train, l_error_cv

def get_error_mertics(clf, X_cv, y_cv):
    d = {}
    #classifier = clf.fit(X_train, y_train)
    predictions = clf.predict(X_cv)
    
    d['hit_rate'] = precision_score(y_cv, predictions)
    d['catch_rate'] = recall_score(y_cv, predictions)
    d['f1'] = f1_score(y_cv, predictions)
    d['accuracy'] = accuracy_score(y_cv, predictions)
    d['fpr'] = sum(y_cv < predictions)/sum(predictions == (y_cv == 1))
    d['predictions'] = predictions
    d['confusion'] = pd.DataFrame(confusion_matrix(y_cv, predictions), columns = ['Not-Pred','Pred'], 
                                  index = ['Not-Fraud','Fraud'])
    d['leakage_rate'] = sum(y_cv > predictions)/sum(predictions <> 1)*10000
    #d['good_decline_rate'] = sum(predictions > 0) 
    return d

#creating dummy features from categorical columns
def get_dummy_cols(df, column_list):
    df_with_dummies = df
    for x in column_list:
        df_with_dummies = df_with_dummies.merge(pd.get_dummies(df[x], prefix = x), 
                                                how = 'inner',left_index = True, right_index = True)
    df = df_with_dummies
    return df

    '''
    dummy_delivery_type = pd.get_dummies(df['delivery_type'], prefix ='delivery_type')
    dummy_market = pd.get_dummies(df['market'], prefix = 'market')
    dummy_client_type = pd.get_dummies(df['client_type'], prefix = 'client_type')
    #df1 = df.merge(dummy_merchant, how = 'inner', left_index = True, right_index = True)
    # replace this code with the code to dummify the merchant category 
    df1 = df
    df2 = df1.merge(dummy_delivery_type, how = 'inner', left_index = True, right_index = True)
    del df1
    df3 = df2.merge(dummy_market, how = 'inner', left_index = True, right_index = True)
    del df2
    df4 = df3.merge(dummy_client_type, how = 'inner', left_index = True, right_index = True)
    del df3
    '''

