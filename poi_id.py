#!/usr/bin/python

import sys
import pickle
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot

def print_stats(features, data):
    #prints stats for features
    print '\n', 'Printing stats for each feature....', '---------------\n',
    for i in range(len(features)):
        print
        print features[i]
        print 'Min', np.min(data[:,i])
        print 'Max', np.max(data[:,i])
        print 'Mean', np.mean(data[:,i])
        print 'Median', np.median(data[:,i])
        print 'Top 5', sorted(data[:,i], reverse=True)[:5]
        print
    return 
    
def calc_ratio_email_poi(dict_entity): #constructs my new feature ratio_email_poi
    from_poi = 0
    to_poi = 0
    to_messages = 0
    from_messages = 0
    
    address = dict_entity['email_address']

 
    if dict_entity['from_poi_to_this_person'] != 'NaN':
        from_poi = dict_entity['from_poi_to_this_person']       
        
    if dict_entity['from_this_person_to_poi'] != 'NaN':
        to_poi = dict_entity['from_this_person_to_poi']
        
        
    if dict_entity['to_messages'] != 'NaN':
        to_messages = dict_entity['to_messages']
        
        
    if dict_entity['from_messages'] != 'NaN':
        from_messages = dict_entity['to_messages']
    
        
    if from_poi + to_poi == 0:
        dict_entity['ratio_email_poi'] = 0
        
    else:
        dict_entity['ratio_email_poi'] = (to_messages + from_messages)  / (from_poi + to_poi)

    return dict_entity
    
def convNaN(feature): #helper to convert NaN's to zeros
    if feature == 'NaN':
        return 0
    else:
        return feature
    
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi". 
features_list = ['poi', 'total_payments', 'bonus', 'expenses', 'from_messages'] # My guessed features

#Here are all the features available. I tried them with SelectKBest by using the list below
'''features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
    'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 
    'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages',  'from_poi_to_this_person', 
    'from_messages', 'from_this_person_to_poi']'''




### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
 
# first here are the total number of records

print len(data_dict), ' entities or people in the data' #146 records
print 'Each entity has', len(data_dict['TOTAL']), 'features' # 21 features

 
data = featureFormat(data_dict, features_list) #cleans the data for easier analysis
#how many POI's in the set?
print sum(data[:,0]), 'persons of interest total' # 18 POI's

# There are a lot of records with incomplete data represented by NaN.
# count the NaN's
# Outputs Total NaNs 1358 out of 3066 data points
nans_count = 0
for i in data_dict:
    for f in data_dict[i]:
        if data_dict[i][f] == 'NaN':
            nans_count += 1
print 'Total NaNs', nans_count, 'out of', 21 * 146, 'data points' 
        

# I constructed a helper function called convNaN to turn NaN's into zeros, please see above


### Task 2: Remove outliers

# First I printed the max, min, mean, and median, top 5 of each feature
# I constructed a function called print_stats. Please see above
 
print_stats(features_list, data) 



# I noticed one extreme outlier for total_payments
# Remove extreme outlier named 'TOTAL' in data_dict

extreme_outlier = ''
for i in data_dict:
    if data_dict[i]['total_payments'] == 309886585.0:
#        print i, data_dict[i] # who is this? -TOTAL is an error
        extreme_outlier = i
del data_dict[extreme_outlier]
  
# reload data for data_dict and print again

data = featureFormat(data_dict, features_list)
print_stats(features_list, data)

#found one more outlier, repeat process

extreme_outlier = ''
for i in data_dict:
    if data_dict[i]['total_payments'] == 103559793.0:
        extreme_outlier = i
        # Removing Kenneth Lay
del data_dict[extreme_outlier]


#Verify financial data by checking to see if the sum of payments equals total payments

list_financials = ['salary', 'deferral_payments', 'loan_advances', 'bonus', 'deferred_income', 'expenses', 
'other', 'long_term_incentive', 'director_fees']

    
for i in data_dict:
    total_finance = 0
    for f in list_financials:
        total_finance += convNaN(data_dict[i][f])
        
    if total_finance != convNaN(data_dict[i]['total_payments']):   
        print i, 'expected $', convNaN(data_dict[i]['total_payments']), 'got $', total_finance
# we have two people who have financials that are incorrect. lets remove them
del data_dict['BELFER ROBERT']
del data_dict['BHATNAGAR SANJAY']




### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

#I am going to implement a new feature, ratio of communication is with poi or ratio_email_poi
# it will be the sum of 'from_poi_to_this_person','from_this_person_to_poi' divided by total emails
# first I need to calculate total emails for each person, then ratio.
# I made a function called calc_ratio_email_poi() Please see above

for i in data_dict:
    data_dict[i] = calc_ratio_email_poi(data_dict[i])

#features_list.append('ratio_email_poi') #Made score slightly worse so commented out
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


#from sklearn.naive_bayes import GaussianNB #starting point
#clf = GaussianNB() #accuracy .82, precision .07, bad recall .014

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier( min_samples_leaf = 1, min_impurity_decrease = .001,
    min_samples_split = 2) #accuracy .74, precision 0.17, recall 0.17 before adding features, tuning

#from sklearn.svm import SVC
#clf = SVC(kernel='rbf', C = 10000., class_weight = 'balanced') # no True Positives or Predicted Positives

#from sklearn.neighbors import KNeighborsClassifier
#clf = KNeighborsClassifier() #no true or predicted Positives at all

#from sklearn.ensemble import RandomForestClassifier #bad recall 0.17
#clf = RandomForestClassifier(n_estimators = 100) #resulted in low recall

#from sklearn.ensemble import AdaBoostClassifier
#clf = AdaBoostClassifier(n_estimators = 50)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# I tried using SelectKBest feature selection here but got unacceptable results 
'''from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
selection_clf = SelectKBest(k = 10)
selection_clf.fit_transform(features_train, labels_train)
print selection_clf.scores_'''


clf.fit(features_train, labels_train) 

#selection_clf.transform(features_test) #applied SelectKBest to test set
print '\n', clf.score(features_test, labels_test), 'score'


from sklearn.metrics import confusion_matrix
print 'Confusion Matrix \n', confusion_matrix(labels_test,clf.predict(features_test))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)