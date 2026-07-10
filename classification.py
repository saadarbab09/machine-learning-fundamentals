import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import tensorflow as tf



#MAKING DATASET
columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha' , 'fDist', 'class']
df =  pd.read_csv('magic04.data', names = columns)
df['class'] = (df['class'] == 'g').astype(int)


# CREATING HISTOGRAMS FOR EACH FEATURE
SHOW_HISTOGRAMS = False

if SHOW_HISTOGRAMS:

    for label in columns[:-1]:
        plt.figure()
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




 #***** (3) LOGISTIC REGRESSION *****
 # this model  uses y = mx + c to derive the sigmoid fuction.
 # which outputs the probability of a sample belonging to a class.
 # the sigmoid function is defined as: 1 / (1 + e^(-z)).


 # IMPLEMENTATION:
# from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()# as default it uses L2 penalty which is quadratic: penalises outliers heavily.
lr_model = lr_model.fit(X_train, y_train) #fit the model to the training data
y_predicted_lr = lr_model.predict(X_test) #predict the class labels for the test
# print(classification_report(y_test, y_predicted_lr)) #print the classification report LR model.


##### LOGISTIC REGRESSION CLASSIFICATION REPORT:
#               precision    recall  f1-score   support

#            0       0.67      0.71      0.69      1338
#            1       0.84      0.81      0.82      2466

#     accuracy                           0.77      3804
#    macro avg       0.75      0.76      0.76      3804
# weighted avg       0.78      0.77      0.77      3804



 #***** (4) SUPPORT VECTOR MACHINE (SVM) *****
 # this model finds the best hyperplane that best divides the classes in the feature space.
 # it also maximizes the margin between the svm hyperplane and support vectors
 # which helps improve generalization and reduce overfitting. 
 #for the feature space that is not linearly separable.
 # it uses kernel trick to map data into a higher dimensional space that is seperable.


 # IMPLEMENTATION:
# from sklearn.svm import SVC
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train) #fit the model to the training data
y_predicted_svm = svm_model.predict(X_test) #predict the class labels for the test
# print(classification_report(y_test, y_predicted_svm))  classification report


##### SVM CLASSIFICATION REPORT:
#               precision    recall  f1-score   support

#            0       0.83      0.79      0.81      1347
#            1       0.89      0.91      0.90      2457

#     accuracy                           0.87      3804
#    macro avg       0.86      0.85      0.85      3804
# weighted avg       0.86      0.87      0.86      3804





#***** (4) NEURAL NETWORK FOR CLASSIFICATION *****

# this model uses multiple layers of interconnected nodes to learn complex patterns in the data.
# it uses activation functions to remove linearity. sigmoid, relu, tanh are some of the activation functions.
# and uses gradient descent backpropagation to update weights for adjusting the model to minimize the loss function.
# we will use the tesorflow library to implement this model.

#IMPLEMENTATION:
## OUR FIRST NN (BASIC PARAMETERS)
# import tensorflow as tf
BASIC_NN = False
if BASIC_NN:

    nn_model = tf.keras.Sequential([
        tf.keras.Input(shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),# start with 32 n layer , define activation func and tell input shape number of features in the input data. 
        tf.keras.layers.Dense(32, activation='relu'), # second layer with 32n
        tf.keras.layers.Dense(1, activation='sigmoid') # condense it to 1 node , with sigmoid for classification.

    ])
# to configure the model we use .compile(), which tells how to:
# update weights, measure errors, what performance metric to report  
    nn_model.compile(optimizer = tf.keras.optimizers.Adam(0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
#optimizer is the algorithm that updates the weights during backpropogation, learning rate alpha = 0.001
#(a smarter version of gradient descent that auto adjusts and updates )
# loss = 'binarycrossentropy' is a loss function to model loss 
# metrics = ['accuracy'] not for training. keras calculates and records the accuracy with each cycle epoch
# to visualise these stats and accuracy well use tensorflow plots
#to plot the loss over each epoch/training cycle:
def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()
#to plot the accuracy over each epoch/training cycle:
def plot_accuracy(history):
    plt.figure()
    plt.plot(history.history['accuracy'], label = 'accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()   
# plotting loss and accuracy side by side :
# 
def plot_history(history):
    # Create a new figure with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ----- Loss -----
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary Crossentropy')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    # ----- Accuracy -----
    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()    


### TRAINING THE MODEL ###
if BASIC_NN:
    history = nn_model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )#nn_model.fit trains the model , and just returns ONE OBEJECT that is history, which stores:
#     loss': [...],
#     'accuracy': [...],
#     'val_loss': [...],
#     'val_accuracy': [...]
# bathsize means intead of taking all the sample togoether take in batches of 32 
    # Training is done in batches of 32 samples. After each batch, the model makes predictions,
# computes the loss, performs backpropagation, and updates the weights.
# This continues until all training samples have been processed once, which completes one epoch.-    
# plot_loss(history) # look at plot loss (1) /plots for neural network performance 
# plot_accuracy(history)# look at plot (1) /plots accuracy for neural network performance
# 
# ##### now we need to see by adjusting hyperparameters which ones give the highest performance
# so we do a GRID SEARCH: we test different parameters to see which one will give best.

#####   GRID SEARCH  #####
def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
    nn_model = tf.keras.Sequential([
            tf.keras.Input(shape=(10,)),
            tf.keras.layers.Dense(num_nodes, activation='relu'),# start with 32 n layer , define activation func and tell input shape number of features in the input data. 
            tf.keras.layers.Dropout(dropout_prob),# randomly slect nodes dont train them to reduce overfitting 
            tf.keras.layers.Dense(num_nodes, activation='relu'), # second layer with 32n
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(1, activation='sigmoid') # condense it to 1 node , with sigmoid for classification.

        ])
    nn_model.compile(optimizer = tf.keras.optimizers.Adam(lr), loss = 'binary_crossentropy', metrics = ['accuracy'])
    history = nn_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    return nn_model, history
# implementting grid search 
grid_search = False
if grid_search:
    least_val_loss = float('inf')
    least_loss_model = None
    epochs = 100
    best_params = None
    best_history = None
    for num_nodes in [16, 32, 64]:
        for dropout_prob in [0, 0.2]:
            for lr in [0.01, 0.005, 0.001]:
                for batch_size in [32, 64, 128]:
                    print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
                    #train models returns model and history 
                    model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
                    #plot_history(history)
                    val_loss, val_accuracy = model.evaluate(X_valid, y_valid)# check each model with new data(valid)
                    if val_loss < least_val_loss:
                        least_val_loss = val_loss
                        least_loss_model = model
                        best_history = history
                        best_params = (num_nodes, dropout_prob, lr, batch_size)
    print(f"Best model: {best_params[0]} nodes, dropout={best_params[1]}, learning rate={best_params[2]}, batch size={best_params[3]}")
    print(f"Lowest validation loss: {least_val_loss:.4f}")
# grid search results :
# Best model: 64 nodes, dropout=0.2, learning rate=0.001, batch size=64
# Lowest validation loss: 0.3045
# accuracy : 0.88

#plot for grid search result(best params, best history):
# best_model, best_history = train_model(
#     X_train,
#     y_train,
#     num_nodes=64,
#     dropout_prob=0.2,
#     lr=0.001,
#     batch_size=64,
#     epochs=100
# )

# plot_history(best_history)# LLOK AT PNG IN PLOTS FOR NN(GRID SEARCH)


#   ******CLASSIFIACATION REPORT FOR LEAST_LOSS MODEL:
# y_predicted_nn = best_model.predict(X_test)
# y_predicted_nn = (y_predicted_nn > 0.5).astype(int).reshape(-1,)
# print(classification_report(y_test, y_predicted_nn))


##### NN (AFTER GRID SEARCH) CLASSIFICATION REPORT:

#               precision    recall  f1-score   support

#            0       0.90      0.73      0.81      1370
#            1       0.86      0.96      0.91      2434

#     accuracy                           0.87      3804
#    macro avg       0.88      0.84      0.86      3804
# weighted avg       0.88      0.87      0.87      3804