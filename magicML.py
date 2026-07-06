import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB



#MAKING DATASET
columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha' , 'fDist', 'class']
df =  pd.read_csv('magic04.data', names = columns)
df['class'] = (df['class'] == 'g').astype(int)


# CREATING HISTOGRAMS FOR EACH FEATURE
for label in columns[:-1]:
    plt.hist(df[df['class']==1][label], color = 'blue', alpha = 0.7, label = 'gamma', density = True)
    plt.hist(df[df['class']==0][label], color = 'red', alpha = 0.7, label = 'hadron', density = True)
    plt.title(label)
    plt.ylabel('Density')
    plt.legend()


#PREPROCESSING DATASET


# SPLITTING DATASET INTO TRAINING, VALIDATION AND TESTING    
shuffled_df = df.sample(frac=1).reset_index(drop=True)
train = shuffled_df.iloc[:int(0.6 * len(df))]
valid = shuffled_df.iloc[int(0.6 * len(df)):int(0.8 * len(df))]
test = shuffled_df.iloc[int(0.8 * len(df)):]



#   ******SCALING DATASET******
#scale for each label is way off that can negatively affect out results
# we need to scale the data, that is relative to the mean and standard deviation of each column
# this function will do exaclty that.
def scale_dataset(dataframe, oversample=False): 
    # x will be a numpy array storing all the feature matrix without class label in 2D
    x = dataframe[dataframe.columns[:-1]].values  
    # y will be a numpy array storing all the class labels in 1D
    y = dataframe[dataframe.columns[-1]].values
# Standardize each feature by subtracting its mean and dividing by its standard deviation,
# resulting in features with mean ≈ 0 and standard deviation ≈ 1.
# val - mean / std : well do this using standardscaler from sklearn library
#from sklearn.preprocessing import StandardScaler

    
    scaler = StandardScaler()
    x = scaler.fit_transform(x) 
# scaleris a StandardScaler object from scikit-learn, which provides methods for scaling.
# fit() learns and stores the mean and standard deviation of each col.
# transform() uses the stored mean & sd to scale.


#THIS IS CLASS BALANCING (EXPLAINED AFTER THE FUNCTION):

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)

#....HSTACK....
# for each sample of x attach the corresponding target y to the end of that row.
#we split the the data into x and y to apply saclling and balancing.
# StandardScaler only works on features (X)
#RandomOverSampler expects X and y separately.
#After preprocessing, we recombine them with:
    data = np.hstack((x, y.reshape(len(y), 1))) # for hstack both arrays need to have same number of rows.
# RETURNS 3: THE COMPLETE DATASET(SCALLED AND BALANCED) AS Data, THE featue & target (x&y).   
    return data, x, y


#******BALANCING THE DATASET(CLASS)******

# print(len(train[train[:, -1] == 1]))  # Gamma
# print(len(train[train[:, -1] == 0]))  # Hadron
#7399 FOR GAMMA AND 4013 FOR HADRON: CLASS IMBALANCE,
# # we can use oversampling to balance the dataset
#IMPORT  imblearn RandomOversampler
# from imblearn.over_sampling import RandomOverSampler 


#ADDED TO THE Scale_dataset function :
# oversample = false this is added to the parameter of the function scale_dataset.
# when calling the scale_dataset func, this will be set to false by default.
#but if we want to balance we need to set it to true.
# ros = RandomOverSampler()
# x, y = ros.fit_resample(x, y)
#  the above 2 lines create a RandomOverSampler object from imblearn.
# fit_resample() identifies the minority class and randomly duplicates its samples.


#CALLING AND SAVING THE PROCESSED DATASET:
train, X_train, y_train = scale_dataset(train, oversample=True)#TRAIN
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)#VALIDATION
test, X_test, y_test = scale_dataset(test, oversample=False)#TEST
# scaledataset() returns 3 values.
# train = data , X_train = x, y_train = y
#updating train, new : X_train and y_train processed feature matrix and target vector.
# print(len(y_train)) 14722
# print(sum(y_train==1))  # Gamma = 7364
# print(sum(y_train==0))  # Hadron = 7364
#oversampling ha sbeen false for validation and test sets, so they remain imbalanced.
#inorder to evaluate the model's performance on real-world data, we should not oversample.






#*****BUILDING THE MODELS*****
 
#***** (1) K-NEAREST NEIGHBORS *****

#this model finds the closest training examples in the feature space and predicts the class 
# based on majority voting.

#IMPLEMENTATION:
# from sklearn.neighbors import KNeighborsClassifier 
knn_model = KNeighborsClassifier(n_neighbors=1) # n means the number of neighbors to consider.
knn_model.fit(X_train, y_train) #fit the model to the training data
y_predicted = knn_model.predict(X_test) #predict the class labels for the test set


# print(y_predicted) model gave : [1 1 1 ... 0 1 1]
# print(y_test) actual was : [1 1 1 ... 0 1 1]


#TO CHECK THE CLASSIFICATION RESULT BY THIS MODEL WE USE THE CLASSIFICATION REPORT.
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_predicted))


#### KNN CLASSIFICATION REPORT:
#                 precision    recall  f1-score   support

#            0       0.77      0.67      0.72      1337
#            1       0.83      0.89      0.86      2467

#     accuracy                           0.81      3804
#    macro avg       0.80      0.78      0.79      3804
# weighted avg       0.81      0.81      0.81      3804



 #***** (2) NAIVE BAYES *****

 #  this model uses the bayes theorem and conditional probability to predict class.
 # takes each feature as independent events, and calculates the probability of each class given the features.
 # uses argmax MAP to select the class with the highest probability., to reduce the chance of misclassification.
 # # P(Class | Features) = (P(Features | Class) × P(Class)) / P(Features)
 # # P(Class | Features) = (P(Class) × ∏ P(Feature_i | Class)) / P(Features)

#IMPLEMENTATION:
 # from sklearn.naive_bayes import GaussianNB  
nb_model = GaussianNB() # intialise the nb_model as an instance of GaussianNB class.
nb_model = nb_model.fit(X_train, y_train) #fit the model to the training data
y_predicted_nb = nb_model.predict(X_test) #predict the class labels for the test set
# print(classification_report(y_test, y_predicted_nb)) #print the classification report for the Naive Bayes model


##### NAIVE BAYES CLASSIFICATION REPORT:
#                 precision    recall  f1-score   support

#            0       0.71      0.41      0.52      1370
#            1       0.73      0.91      0.81      2434

#     accuracy                           0.73      3804
#    macro avg       0.72      0.66      0.67      3804
# weighted avg       0.73      0.73      0.71      3804
    
