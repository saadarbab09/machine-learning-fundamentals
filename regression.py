import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler 
import copy
import seaborn as sns
import tensorflow as tf
from sklearn.linear_model import LinearRegression
# DATA PREPROCESSING
dataset_cols = ['bike_Count', 'hour', 'temp', 'humidity', 'wind', 'visibility', 'dew_pt_temp', 'radiation', 'rain', 'snow', 'functional' ]
df = pd.read_csv('SeoulBikeData.csv', encoding='latin1').drop(['Date', 'Holiday', 'Seasons'], axis=1)

df.columns = dataset_cols
df['functional'] = (df['functional'] == 'Yes').astype(int)
# print(df["functional"].value_counts())
#keep all where hours = 12
df = df[df['hour'] == 12]
#now drop the hours entire collumn
df = df.drop(['hour'], axis = 1 )
# print(df.head())


#plotting graphs for all features against bike count
starting_graphs = False
if starting_graphs == True:
    for label in df.columns[1:]:
        plt.scatter(df[label], df['bike_Count'])
        plt.title(label)
        plt.ylabel('Bike count at Noon')
        plt.xlabel(label)
        # plt.show()

# look at the scatter plots, not much relation forund in wind visibility and functional, hence dropped 

df =  df.drop(['wind', 'visibility', 'functional'], axis = 1 )
# print(df.head())
## DATA AFTER PREPROCESSING 
#      bike_Count  temp  humidity  dew_pt_temp  radiation  rain  snow
# 12          449   1.7        23        -17.2       1.11   0.0   0.0
# 36          479   4.3        41         -7.8       1.09   0.0   0.0
# 60          333   5.8        85          3.4       0.43   0.0   0.0
# 84          393  -0.3        38        -12.9       1.11   0.0   0.0
# 108         321  -2.3        25        -19.7       0.00   0.0   0.0
#              NOW SPLIT INTO TRAIN, VALID AND TEST
shuffled_df = df.sample(frac=1).reset_index(drop=True)
train = shuffled_df.iloc[:int(0.6 * len(df))]
val = shuffled_df.iloc[int(0.6 * len(df)):int(0.8 * len(df))]
test = shuffled_df.iloc[int(0.8 * len(df)):]
# unlike the classiifer script where the last collumn will be the target, we can choose the,
# what the the target and feature colluns can be.


#quick recap for the scale data set func used in classifier script:
# (1) data set given, split in train valid and test
# (2) x and y chosen from coll index x being all untill last and y just last.
# (3) apply scaling, oversampling and balancing to x and y , and then combine x&y using hstack
# (4) x was just the feature matrix with out the y 
# (5) y was just the target vector 
# (6) the func returned 3 things new df, x and y 
 

 #NOW NEW FUNC
def get_xy(dataframe, y_label, x_labels= None):
    #dataframe : Which df to process and extract
    #y_label : the name of the target collumn 
    #x_label : which col or cols u want as feature 
    #x_label= None: if no such is provided, use every column except the target as features.


    dataframe = copy.deepcopy(dataframe)# creates a completely separate copy
    #So any changes made inside the function do not accidentally modify the original:
    # Train , val and test
    # Original train → remains untouched
    # Copied train   → used inside the function


    if x_labels is None: # **CASE 1**: No specific feature names given
        # means if x_labels is None 
        # then we need all the collumns as feature except the given target 
        #so:
        X = dataframe[[c for c in dataframe.columns if c != y_label]].values
        # x becomes the feature matric of all the collumns except the given y col
    else:

        #***CASE 2 *** this has cases inside 
        # if x col is given: branches out to if 1 given or more than 1 
        # if 1 is given :
        if len(x_labels)==1:
            X = dataframe[x_labels[0]].values.reshape(-1, 1)
            #x_labels[0] x_label can be a list of all the names of the columns wanted 
            #in this case there is only 1 so the 0th index works 
            #df[name of col] returns a pandas sereis
            #df[name of col].VALUES returns a numpy array 
            # rehape(-1,1)(r,c) columns =1 return 1 col, and -1 means figure out the num for rows


            #*** CASE 2 SUB CASE 2:
            # more than 1 feature given :
        else:
            X = dataframe[x_labels].values
    y = dataframe[y_label].values.reshape(-1, 1)
    data = np.hstack((X, y))

    return data, X, y

# USING THE GET_XY FUNC:
# Only USING 1 FEATURE **** temp****
_, X_train_temp, y_train_temp = get_xy(train, 'bike_Count', x_labels=['temp'])
_, X_val_temp, y_val_temp = get_xy(val, 'bike_Count', x_labels=['temp'])
_, X_test_temp, y_test_temp = get_xy(test, 'bike_Count', x_labels=['temp'])
# _ means to ignore and not store the returning value so here the data variable is being ignored


#**** BUILDING THE MODEL :
#*** SIMPLE LINEAR REG(TEMP):
temp_reg = LinearRegression()
temp_reg.fit(X_train_temp, y_train_temp)
# print(temp_reg.coef_, temp_reg.intercept_)
# y = mx + c 
# m = [[19.33158465]] 
# c = [373.61359616]


# **** Rsquared***
# print(temp_reg.score(X_test_temp, y_test_temp))
#  0.33548393109996355
# .score() returns R² (R-squared).
# 0  → Very poor model
# 1  → Perfect model
# Using only temperature, the model can explain about 38.3% of the variation in bike rentals.
# r^2 = 1-(rss/tss) rss error wrt to line of best, tss error wrt avg of y
# the smaller the rss is wrt to tss then rss/tss will be samller tf score will be closer to 1


# plot for SIMPLE LINEAR REG(TEMP):
simple_lin_reg_plot = False
if simple_lin_reg_plot:
    plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
    # Generate/invent  100 evenly  values from -20°C to 40°C random not from dataset
    x = tf.linspace(-20, 40, 100)
    # Predict bike counts for those values generated using the y=mx+c and plot the regression line
    #building the line of best fit
    plt.plot(
        x,
        temp_reg.predict(np.array(x).reshape(-1, 1)),  # Reshape because predict() expects a 2D feature matrix
        label="Fit",
        color="red",
        linewidth=3
    )
    plt.legend()
    plt.title("Bikes vs Temp")
    plt.ylabel("Number of bikes")
    plt.xlabel("Temp")
    plt.show()




#  MULTIPLE LINEAR REGRESSION 
# now all the colluns as features except the bike count so exclude index 0
#x_labels= df.columns[1:])

_, X_train_all, y_train_all = get_xy(train, 'bike_Count', x_labels= df.columns[1:])
_, X_val_all, y_val_all = get_xy(val, 'bike_Count', x_labels= df.columns[1:])
_, X_test_all, y_test_all = get_xy(test, 'bike_Count', x_labels= df.columns[1:])


#MAKING THE MODEL
all_reg = LinearRegression()
all_reg.fit(X_train_all, y_train_all)


#Rsquared score
# print(all_reg.score(X_train_all, y_train_all)) 
# 0.5015708540856685





#**** REGRESSION WITH NEURAL NET****



#ONLY TEMP FIRST:

#we need to first scale the data for the neural net 
# for this we will use tensorflow normalization layer(subtract mean divide by sd)
temp_normalizer = tf.keras.layers.Normalization(#initializing the layer
    input_shape = (1,), axis=None 
    ) 
#input_shape telling the layer number of features that will come (1,):each row has 1 value
#axis = none treat all the input as one group calculate 1 mean 1 sd
#uptill now just this layer has built nothing else only knows what to expect(parameters)
temp_normalizer.adapt(X_train_temp.reshape(-1))#feeding the data to the layer, learning the mean,sd
#similar to fit, learn its mean and sd , adapt accepts 1d thats rehape to (-1)
#normalization does not happen here , it has made a layer and then learned the stats

#BUILDING THE MOODEL
##Defining the model, the structure/foundation
#sequential : data will go through these layers 
simple_nn = False
if simple_nn:
    temp_nn_model= tf.keras.Sequential([
        temp_normalizer, # layer 1 the normalizer we initiated 
        tf.keras.layers.Dense(1)# layer 2: creates 1 neuron (we only need 1 )

    ]) 
    #when this model is called then normalization will take place.

    #NOW CONFIGURING THE MODEL compile()
    temp_nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),# setting the optimizer 
        loss='mean_squared_error'# as this is predicting continuos mean squared erros is good loss func
    )
    #NOW start the  TRAINING THE MODEL FIT()
    #FIT 1)TRAINs the model and 2) returns the history object 
    history = temp_nn_model.fit(
        X_train_temp.reshape(-1), y_train_temp,# reshaping x train temp cuz normalization expects this shape
        #y tain is the actual to compare with and update weights and see performance 
        verbose=1,
        epochs=1000,
        #Train the model 1000 times over the entire training dataset.
        validation_data=(X_val_temp, y_val_temp)
        #after each epoch tf checks on val the performance of the model 
        #HISTORY IS STORING LOSS AND VAL LOSS 
        #remember backpropgation and weigh tupdates only happer in training data set after each batch not val
        #bathes in model --> predict --> backpropogation -> batches until whole ds(1 epoch)-->checking on valid(entire)
    )
# plot for nn regression model 
def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'], label = 'loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()  

# plot_loss(history)

simple_lin_reg_plot_2 = False
if simple_lin_reg_plot_2:
    plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
    # Generate/invent  100 evenly  values from -20°C to 40°C random not from dataset
    x = tf.linspace(-20, 40, 100)
    # Predict bike counts for those values generated using the y=mx+c and plot the regression line
    #building the line of best fit
    plt.plot(
    x,
    temp_nn_model.predict(np.array(x).reshape(-1, 1)),  # Reshape because predict() expects a 2D feature matrix
    label="Fit",
    color="red",
    linewidth=3
    )
    plt.legend()
    plt.title("Bikes vs Temp")
    plt.ylabel("Number of bikes")
    plt.xlabel("Temp")
    plt.show()

#*** ACTUAL NEURAL NET
#SETUP:
nn_1_feature = False
if nn_1_feature:
    nn_model= tf.keras.Sequential([
            temp_normalizer, # layer 1 the normalizer we initiated 
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1),

        ]) 
    #CONFIG:
    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),# setting the optimizer 
        loss='mean_squared_error'
    )
    #TRAINING:
    history = nn_model.fit(
        X_train_temp, y_train_temp,
        validation_data = (X_val_temp, y_val_temp),
        verbose = 1,
        epochs = 100    
    )
# plot_loss(history)
simple_lin_reg_plot_3 = False
if simple_lin_reg_plot_3:
    plt.scatter(X_train_temp, y_train_temp, label="Data", color="blue")
    # Generate/invent  100 evenly  values from -20°C to 40°C random not from dataset
    x = tf.linspace(-20, 40, 100)
    # Predict bike counts for those values generated using the y=mx+c and plot the regression line
    #building the line of best fit
    plt.plot(
    x,
    nn_model.predict(np.array(x).reshape(-1, 1)),  # Reshape because predict() expects a 2D feature matrix
    label="Fit",
    color="red",
    linewidth=3
    )
    plt.legend()
    plt.title("Bikes vs Temp")
    plt.ylabel("Number of bikes")
    plt.xlabel("Temp")
    plt.show()


nn_with_multiple_features = False
if nn_with_multiple_features: 
    all_normalizer = tf.keras.layers.Normalization(#initializing the layer
        input_shape = (6,), axis=-1 
        ) 
    all_normalizer.adapt(X_train_all)

    nn_model= tf.keras.Sequential([
                all_normalizer, # layer 1 the normalizer we initiated 
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1),

            ]) 
    nn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),# setting the optimizer 
            loss='mean_squared_error'
        )
    history = nn_model.fit(
            X_train_all, y_train_all,
            validation_data = (X_val_all, y_val_all),
            verbose = 1,
            epochs = 100    
        )
    plot_loss(history)