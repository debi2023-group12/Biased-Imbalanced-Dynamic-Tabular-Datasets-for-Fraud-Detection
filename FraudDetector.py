from aequitas.group import Group  # Aequitas is a package for Fairness evaluation
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score,accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import skew


# imports for neural network
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
from keras import backend as K

# Oversampling and under sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE,BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter

from pycaret.classification import setup, create_model, plot_model, tune_model

#making Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno # Missing data visualization
import association_metrics as am # correlation for categorical 


#Income Category:
def income_category(row):
    if row['income'] < 0.3:
        return "Low income"
    elif row['income'] < 0.7:
        return "Medium income"
    else:
        return "High income"

    
#Age Group
def age_group(row):
    age = row['customer_age']
    
    if age >= 0 and age <= 17:
        return "Child/Teenager"
    elif age >= 18 and age <= 35:
        return "Young Adult"
    elif age >= 36 and age <= 55:
        return "Middle-Aged"
    else:
        return "Senior Citizen"

    
#Application Velocity combine velocity_6h & velocity_24h features
def application_velocity(row):
    velocity_6h = row['velocity_6h']
    velocity_24h = row['velocity_24h']
    
    average_velocity = (velocity_6h + velocity_24h ) / 2
    
    if average_velocity < 3000:
        return "Low Velocity"
    elif average_velocity < 6000:
        return "Moderate Velocity"
    else:
        return "High Velocity"

    
#contact validity combine phone_home_valid & phone_mobile_valid
def contact_validity(row):
    phone_home_valid = row['phone_home_valid']
    phone_mobile_valid = row['phone_mobile_valid']
    
    if phone_home_valid or phone_mobile_valid:
        return 1  # At least one contact phone or mobile has a number
    else:
        return 0  # Both contact phones and mobiles have no number
    
def plot_all_roc(fprs_list, tprs_list, model_names):

    # Plot all ROC curves together in one figure
    plt.figure(figsize=(10, 6))
    for fprs, tprs, NameModel in zip(fprs_list, tprs_list, model_names):
        plt.plot(fprs, tprs, label=NameModel)

def get_fairness_metrics(
    y_true, y_pred, groups, FIXED_FPR=0.05):
    g = Group()
    aequitas_df = pd.DataFrame(
        {"score": y_pred,
        "label_value": y_true,
        "group": groups}
    )
    # Use aequitas to compute confusion matrix metrics for every group.
    disparities_df = g.get_crosstabs(aequitas_df, score_thresholds={"score": [FIXED_FPR]})[0]
    
    # Predictive equality is the differences in FPR (we use ratios in the paper)
    predictive_equality = disparities_df["fpr"].min() / disparities_df["fpr"].max()

    return predictive_equality, disparities_df


# plot the false-positive rate of a model compared to the true-positive rate (ROC-Curves)
def plot_roc(fpr, tpr, NameModel):
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of '+NameModel)
    plt.legend()
    plt.show()

def f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

# --- Two currently unused metrics ---
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def create_DNN(num_features, dropout_rate=0.5, neurons=128, learning_rate=1e-2):
    model = keras.Sequential([
        keras.layers.BatchNormalization(input_shape=[num_features]),
        keras.layers.Dense(neurons, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(neurons//2, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(neurons//2, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(neurons//4, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def compile_DNN(model):
    metrics = [
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        f1, 
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(1e-2),
        loss="binary_crossentropy",
        metrics=metrics
    )
    
    return model

# 
def train_DNN(model, X_train, y_train):
    # Use EarlyStopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
        mode='max'
    )
    
    # Calculate the class wheights for the model, improves predictive equality
    class_weights = {0: 1., 1: np.sum(y_train == 0) / np.sum(y_train == 1)}
    
    hist = model.fit(
        X_train, y_train, 
        class_weight=class_weights,batch_size=512,
        epochs=100, # set lower if you only want to train for short period to get approximat results
        callbacks=[early_stopping],
        verbose=1,
        validation_split=0.1 # Use 10% of training set as validation for EarlyStopping
    )
    # return the training history for possible visualization
    return hist

class Bank_Account_Fraud_Detection:


    def __init__(self, dataset_path='data/'):
        
        # Setting the data folder path
        self.path = dataset_path

        # Initializing variables
        self.dataset = pd.DataFrame()
        self.preprocessed = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.splitted = 0
        self.one_hot = 0
        self.std_scale = 0
        self.model = None
        self.groups = None

    # Function for loading the dataset
    def load_data(self, file_name='Base'):
        
        if file_name in ['Base', 'Variant I', 'Variant II', 'Variant III', 'Variant IV', 'Variant V']:
            self.dataset = pd.read_csv(self.path+file_name+'.csv')

            return self.dataset
        
        else:
            print("This is not a valid file name! Please select from the following filenames:")
            print('1. Base\n2. Variant I\n3. Variant II\n4. Variant III\n5. Variant IV\n6. Variant V')

    # Function to split the dataset into train and test sets
    def train_test_split(self):

        if self.dataset.empty:
            self.dataset = self.load_data()
            self.preprocessed = self.preprocess()
            
        if self.splitted == 1:
            pass
        
        else:
            
            X = self.preprocessed.drop(['fraud_bool'], axis=1)
            y = self.preprocessed['fraud_bool']
    
            # Train test split by 'month', month 0-5 are train, 6-7 are test data as proposed in the paper
            self.X_train = X[X['month']<6]
            self.X_test = X[X['month']>=6]
            self.y_train = y[X['month']<6]
            self.y_test = y[X['month']>=6]
            
            self.X_train.drop('month', axis=1, inplace=True)
            self.X_test.drop('month', axis=1, inplace=True)
            
            try:
                self.groups = (self.X_test["customer_age"] > 50).map({True: ">50", False: "<=50"}) 
            except:
                pass
            
            self.splitted = 1


    # Function to one-hot-encode the categorical features
    def one_hot_encoder(self):

        if self.X_train.empty:
            self.train_test_split()
            
        if self.one_hot == 1:
            pass
        
        else:

            s = (self.X_train.dtypes == 'object') # list of column-names and wether they contain categorical features
            object_cols = list(s[s].index) # All the columns containing these features
    
    
            ohe = OneHotEncoder(sparse=False, handle_unknown='ignore') # ignore any features in the test set that were not present in the training set
    
            # Get one-hot-encoded columns
            ohe_cols_train = pd.DataFrame(ohe.fit_transform(self.X_train[object_cols]))
            ohe_cols_test = pd.DataFrame(ohe.transform(self.X_test[object_cols]))
    
            # Set the index of the transformed data to match the original data
            ohe_cols_train.index = self.X_train.index
            ohe_cols_test.index = self.X_test.index
    
            # Remove the object columns from the training and test data
            num_X_train = self.X_train.drop(object_cols, axis=1)
            num_X_test = self.X_test.drop(object_cols, axis=1)
    
            # Concatenate the numerical data with the transformed categorical data
            self.X_train = pd.concat([num_X_train, ohe_cols_train], axis=1)
            self.X_test = pd.concat([num_X_test, ohe_cols_test], axis=1)
    
            # Newer versions of sklearn require the column names to be strings
            self.X_train.columns = self.X_train.columns.astype(str)
            self.X_test.columns = self.X_test.columns.astype(str)
            
            self.one_hot = 1

    
    def standard_scaler(self):

        if self.X_train.empty:
            self.train_test_split()

        if self.std_scale == 1:
            pass
        
        else:

            scaler = StandardScaler()
    
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)
            
            self.std_scale = 1

    def fit_baseline(self, model='LR'):

        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()

        predictions = []

        if model == 'LR':
            self.model = LogisticRegression(
                class_weight='balanced'
            )
            self.model.fit(self.X_train, self.y_train)
            predictions = self.model.predict_proba(self.X_test)[:, 1]
            _, _ = self.evaluate(predictions, 'Logistic Regression Baseline')
            return predictions
        
        elif model == 'XGB':
            self.model = xgb.XGBClassifier(
            tree_method='gpu_hist', gpu_id=0, 
            scale_pos_weight=89.67005
            )
            self.model.fit(self.X_train, self.y_train)
            predictions = self.model.predict_proba(self.X_test)[:, 1]
            _, _ = self.evaluate(predictions, 'XGBoost Baseline')
            return predictions
        
        elif model == 'RF':

            self.model = RandomForestClassifier(class_weight='balanced')
            self.model.fit(self.X_train, self.y_train)
            predictions = self.model.predict_proba(self.X_test)[:, 1]
            _, _ = self.evaluate(predictions, 'Random Forest Baseline')
            return predictions
        
    
        elif model == 'DNN':

            self.model = create_DNN(self.X_train.shape[1])

            self.model = compile_DNN(self.model)
            
            history = train_DNN(self.model, self.X_train, self.y_train)

            predictions = self.model.predict(self.X_test).flatten()

            _, _ = self.evaluate(predictions, 'DNN Baseline')
            return predictions

        else:
            print("This is not a valid model name! Please select from the following filenames:")
            print('1. LR (aka. Logistic Regression)\n2. XGB (aka. XGBoost)\n3. RF (aka. Random Forest)\n4. DNN (aka. Deep Neural Net)')

    

    # Create lists to store the results for each technique
    def evaluate(self, predictions, NameModel, FIXED_FPR = 0.05):
        fprs, tprs, thresholds = roc_curve(self.y_test, predictions)
        plot_roc(fprs, tprs, NameModel)
        tpr = tprs[fprs<FIXED_FPR][-1]
        fpr = fprs[fprs<FIXED_FPR][-1]
        threshold = thresholds[fprs<FIXED_FPR][-1] 
        print("AUC:", roc_auc_score(self.y_test, predictions))
        to_pct = lambda x: str(round(x, 4) * 100) + "%"
        print("TPR: ", to_pct(tpr), "\nFPR: ", to_pct(fpr), "\nThreshold: ", round(threshold, 2))
        predictive_equality, disparities_df = get_fairness_metrics(self.y_test, predictions, self.groups, FIXED_FPR)
        print("Predictive Equality: ", to_pct(predictive_equality))
        print('-------------'*10)   
        return fprs, tprs


    def preprocess(self, option='baseline', fet_eng=False):

        if self.dataset.empty:
            self.dataset = self.load_data()

        if option == 'baseline':
            self.preprocessed = self.dataset.drop(['device_fraud_count'], axis=1, errors='ignore') 
            self.train_test_split()
            self.one_hot_encoder()
            self.standard_scaler()

        
        elif option == 'option1':
            self.preprocessed = self.dataset.drop(['device_fraud_count'], axis=1, errors='ignore')
            # Define the columns where you want to replace -1 with NaN
            columns_to_replace = ['session_length_in_minutes', 'device_distinct_emails_8w',
                                'bank_months_count','prev_address_months_count','current_address_months_count']

            # Replace -1 with NaN in the specified columns
            self.preprocessed[columns_to_replace] = self.preprocessed[columns_to_replace].replace(-1, np.nan)

            # Replace negative values with NaN in a specific column
            self.preprocessed['intended_balcon_amount'] = self.preprocessed['intended_balcon_amount'].mask(self.preprocessed['intended_balcon_amount'] < 0, np.nan)
            
            missing_values_count = self.preprocessed.isnull().sum()
            

            # Filter only the columns with missing values
            missing_df = missing_values_count[missing_values_count > 0]

            # Sort the DataFrame by the missing values count in descending order
            missing_df = missing_df.sort_values(ascending=False)
            
            print(missing_df.index)
            # Create a bar plot to visualize the count of missing values in each column
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 30))
            sns.barplot(x=missing_df.index, y=missing_df.values, palette='viridis', ax=ax1)
            ax1.set_xticklabels(labels=missing_df.index, rotation=90)
            ax1.set_xlabel('Column Name')
            ax1.set_ylabel('Number of Missing Values')
            ax1.set_title('Missing Values Count in Columns with Missing Values')
            
            # drop the columns that exceed 70% of null values
            self.preprocessed = self.preprocessed.drop(['prev_address_months_count','intended_balcon_amount'], axis=1) 

            msno.matrix(self.preprocessed, ax=ax2)

            msno.heatmap(self.preprocessed, ax=ax3)
            fig.tight_layout()
            plt.show()

            #fill the nan values of 'session_length_in_minutes' & 'device_distinct_emails_8w' with mean
            columns_to_fill = ['session_length_in_minutes', 'device_distinct_emails_8w']
            self.preprocessed[columns_to_fill] = self.preprocessed[columns_to_fill].fillna(self.preprocessed[columns_to_fill].mean())

            #Handling missing value with Imputer
            # Select the columns with missing values
            column_name = ['current_address_months_count','bank_months_count']

            # Create an IterativeImputer object to perform regression imputation
            reg_imputer = IterativeImputer(random_state=0)

            # Perform regression imputation
            self.preprocessed[column_name] = reg_imputer.fit_transform(self.preprocessed[column_name])

            # Select columns with object data type
            object_columns = self.preprocessed.select_dtypes(include='object').columns

            # Convert object columns to category data type
            self.preprocessed[object_columns] = self.preprocessed[object_columns].astype('category')

            # there are high correlation between velocity_4w & month with 85%
            self.preprocessed = self.preprocessed.drop(['velocity_4w'], axis=1) 

            if fet_eng:

                self.preprocessed['age_category'] = self.preprocessed.apply(age_group, axis=1)
                self.preprocessed['income_category'] = self.preprocessed.apply(income_category, axis=1)
                self.preprocessed['application_velocity'] = self.preprocessed.apply(application_velocity, axis=1)
                self.preprocessed['contact_validity'] = self.preprocessed.apply(contact_validity, axis=1)

                # drop the columns that I make feature engineering on it 
                self.preprocessed = self.preprocessed.drop(['income','velocity_6h','velocity_24h','phone_home_valid','phone_mobile_valid'], axis=1)

            #drop the column that has high correlation between proposed_credit_limit & credit_risk_score is 61%
            self.preprocessed = self.preprocessed.drop(['proposed_credit_limit'], axis=1) 

            cat_columns = self.preprocessed.select_dtypes(include='object').columns
            
            self.preprocessed = pd.get_dummies(self.preprocessed, columns=cat_columns, drop_first=True)
            self.train_test_split()
            self.one_hot_encoder()        
            self.standard_scaler()
            self.X_train.drop(['customer_age'], axis=1, inplace=True)
            self.X_test.drop(['customer_age'], axis=1, inplace=True)

        elif option == 'option2':
            self.preprocessed = pd.get_dummies(self.dataset)
            columns_to_transform = ['days_since_request', 'zip_count_4w', 'proposed_credit_limit']
            self.EDA('skewness')
            # Apply natural logarithm transformation to specified columns
            self.preprocessed[columns_to_transform] = np.log1p(self.preprocessed[columns_to_transform])
            # Drop Highly Correlated Features
            self.preprocessed = self.preprocessed.drop('payment_type_AA', axis=1)
            self.train_test_split()


    def EDA(self, insight):

        if self.dataset.empty:
            self.dataset = self.load_data()

        if insight == 'frauds':
            # Count the number of non-frauds and frauds
            fraud_counts = self.dataset['fraud_bool'].value_counts()

            # Define Seaborn color palette to use
            #palette = sns.color_palette("dark:#5A9_r")
            palette_color = sns.color_palette('bright')
            # Plotting the pie chart
            labels = ['Non-Fraud', 'Fraud']
            explode = (0,0.4)  # To create a separation effect for the 'Fraud' slice

            plt.pie(fraud_counts, labels=labels, colors=palette_color, explode=explode, autopct='%1.1f%%')
            plt.title('Fraud vs Non-Fraud Distribution')
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

            plt.show()
        
        elif insight == 'missing_counts':
            
            if self.preprocessed.empty:
                missing_values_count = self.dataset.isnull().sum()
            
                print(missing_values_count)
            
            else:
                
                missing_values_count = self.preprocessed.isnull().sum()
            
                print(missing_values_count)
        
        elif insight == 'missingno':
            
            if self.preprocessed.empty:
                msno.matrix(self.dataset)
            
                plt.show
            
            else:
                msno.matrix(self.preprocessed)
            
                plt.show

        elif insight == 'correlations':
            
            if self.preprocessed.empty:
            
                object_columns = self.dataset.select_dtypes(include='object').columns
    
                # Convert object columns to category data type
                self.dataset[object_columns] = self.dataset[object_columns].astype('category')
                # Initialize a CamresV object using you pandas.DataFrame
                cramersv = am.CramersV(self.dataset) 
                # will return a pairwise matrix filled with Cramer's V, where columns and index are 
                # the categorical variables of the passed pandas.DataFrame
                cramersv.fit()
    
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 15))
                ax1.set_title('Correlation between categorial features')
                sns.heatmap(cramersv.fit(), ax=ax1, annot=True)
                
                numeric_columns = self.dataset.select_dtypes(include=['float64', 'int64'])
    
                # Compute the correlation matrix for numeric columns
                correlation_matrix = numeric_columns.corr()
    
                ax2.set_title('Correlation between numerical features')
    
                # Plot the correlation matrix as a heatmap
                sns.heatmap(correlation_matrix, cmap=sns.color_palette("flare", as_cmap=True), annot=True,annot_kws={"size": 9}, ax=ax2)
    
    
                # correlation check 
                cor = self.dataset.corr()
                mask = np.triu(np.ones_like(cor))
                heatmap = sns.heatmap(cor, mask=mask, annot=True, cmap=plt.cm.viridis, annot_kws={"fontsize": 6}, fmt=".2f", ax=ax3)  # Set fmt to ".2f" to display 2 decimals
                plt.show()
            
            
            else:
                try:
            
                    object_columns = self.preprocessed.select_dtypes(include='object').columns
        
                    # Convert object columns to category data type
                    self.preprocessed[object_columns] = self.preprocessed[object_columns].astype('category')
                    # Initialize a CamresV object using you pandas.DataFrame
                    cramersv = am.CramersV(self.preprocessed) 
                    # will return a pairwise matrix filled with Cramer's V, where columns and index are 
                    # the categorical variables of the passed pandas.DataFrame
                    cramersv.fit()
        
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 15))
                    ax1.set_title('Correlation between categorial features')
                    sns.heatmap(cramersv.fit(), ax=ax1, annot=True)
                    
                except:
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 15))
                    
                numeric_columns = self.preprocessed.select_dtypes(include=['float64', 'int64'])
    
                # Compute the correlation matrix for numeric columns
                correlation_matrix = numeric_columns.corr()
    
                ax1.set_title('Correlation between numerical features after preprocessing')
    
                # Plot the correlation matrix as a heatmap
                sns.heatmap(correlation_matrix, cmap=sns.color_palette("flare", as_cmap=True), annot=True,annot_kws={"size": 9}, ax=ax1)
    
    
                # correlation check 
                cor = self.preprocessed.corr()
                mask = np.triu(np.ones_like(cor))
                heatmap = sns.heatmap(cor, mask=mask, annot=True, cmap=plt.cm.viridis, annot_kws={"fontsize": 6}, fmt=".2f", ax=ax2)  # Set fmt to ".2f" to display 2 decimals
                plt.show()
                
                
        elif insight == 'skewness':
            num_cols = [n for n in self.dataset.loc[:, :'month'].columns if
                  pd.to_numeric(self.dataset[n], errors='coerce').notna().all() and
                  self.dataset[n].min() >= 0 and
                  self.dataset[n].dtype in ['int64', 'float64']]
            plt.figure(figsize=(12, 8)) 
            skew_features = self.dataset[num_cols].apply(lambda x: skew(x))
            skew_features = skew_features[skew_features > 0.5].sort_values(ascending=False)
            ax = sns.barplot(x=skew_features.index, y=skew_features.values, color='orange')  
            ax.set_ylabel('', fontsize=20)  
            ax.set_xlabel('', fontsize=20)  
            ax.tick_params(axis='both', labelsize=15)  
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=15)  
            ax.axhline(y=1, color='red', linestyle='--', linewidth=3)
            ax.set_title('Skewness', ha = 'center', weight='bold', fontsize=24)
            ax.text(0.01, 1.1, 'Threshold', color='red', transform=ax.transAxes, fontsize=15, weight='bold') 
            sns.despine()
            plt.gca().set_facecolor('white') 
            plt.show()
    
        elif insight == 'description':
            self.dataset.describe(include='all')
        
        elif insight == 'info':
            self.dataset.info()
        
        else:
            print("This is not a valid insight! Please select from the following insights:")
            print('1. frauds\n2. missing_counts\n3. missingno\n4. correlations\n5. skewness\n6. description\n7. info')


    
    def handle_imbalance(self, method):

        if self.X_train.empty:
            self.train_test_split()

        if method == 'NearMiss':
            nearmiss = NearMiss(version=3)
            self.X_train, self.y_train = nearmiss.fit_resample(self.X_train, self.y_train)
        
        elif method == 'Random-Undersampling':
            under = RandomUnderSampler(sampling_strategy=1) 
            self.X_train, self.y_train = under.fit_resample(self.X_train, self.y_train)

        elif method == 'Random-Oversampling':
            ros = RandomOverSampler(sampling_strategy='minority')
            self.X_train, self.y_train = ros.fit_resample(self.X_train, self.y_train)
        
        elif method == 'SMOTE':
            smote = SMOTE(random_state=42)
            self.X_train, self.y_train= smote.fit_resample(self.X_train, self.y_train)

        elif method == 'Borderline-SMOTE':
            bsmote = BorderlineSMOTE(random_state = 101, kind = 'borderline-1')
            self.X_train, self.y_train = bsmote.fit_resample(self.X_train, self.y_train)
        
        elif method == 'Random':
            self.preprocessed = self.preprocessed.sample(n = 400000,random_state=42)
            self.train_test_split()

        else:
            print('That is not a valid method! Please choose Please select from the following methods:')
            print('1. NearMiss\n2. Random-Undersampling\n3. Random-Oversampling\n4. SMOTE\n5. Borderline-SMOTE\n6. Random')
    
    def tune_baseline_LR(self, C=[0.01, 0.1, 1, 10], penalty=['l1', 'l2', 'elasticnet'],
                         solver=['lbfgs', 'saga', 'newton-cg', 'sag']):
        
        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()

        # Define the model
        lr_model_tuned = LogisticRegression(class_weight={0:89.67005,1:10.32995}, max_iter= 500)

        # Define the hyperparameters to tune
        param_grid = {
            'C': C,
            'penalty': penalty,
            'solver': solver
        }

        # Perform random search with cross-validation
        self.model = RandomizedSearchCV(lr_model_tuned, param_grid, cv=5,n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)

        # Get the best hyperparameters
        best_params = self.model.best_params_
        print("Best Hyperparameters:", best_params)

        # Use the best model for predictions
        self.model = self.model.best_estimator_
        predictions = self.model.predict_proba(self.X_test)[:, 1]
        _, _ = self.evaluate(predictions, 'Best Logistic Regression Model After Hyperparameter Tuning')
        return predictions
        
    def tune_baseline_XGB(self, n_estimators=[100, 500, 1000],
                            max_depth=[ 5,7,9,11],
                            learning_rate=[0.01, 0.05, 0.1, 0.2],
                            gamma=[0.1, 0.5, 1, 3]):
        
        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()
            
        # Define the model
        #changed tree_method='auto', del gpu_id=0
        model_xgb_tuned = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, scale_pos_weight=89.67005)

        # Define the hyperparameters to tune
        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'gamma': gamma
        }

        # Perform randomized search with cross-validation
        self.model = RandomizedSearchCV(model_xgb_tuned, param_distributions=param_grid, n_iter=10, cv=5)
        self.model.fit(self.X_train, self.y_train)

        # Get the best hyperparameters
        best_params = self.model.best_params_
        print("Best Hyperparameters:", best_params)

        # Use the best model for predictions
        self.model = self.model.best_estimator_
        predictions = self.model.predict_proba(self.X_test)[:, 1]

        # Evaluate the model
        _, _ = self.evaluate(predictions, 'Best XGBoost Model After Hyperparameter Tuning')
        return predictions
        
    def tune_basline_RF(self, n_estimators=[100, 500],
                    max_depth=[2, 5, 10],
                    max_features=['sqrt', 'log2']):
    
        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()
            
            
        # Define the model
        rf_tuned_model = RandomForestClassifier()

        # Define the hyperparameters to tune
        param_grid = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'max_features': max_features
        }

        # Perform grid search with cross-validation
        self.model = RandomizedSearchCV(rf_tuned_model, param_grid, cv=5,n_jobs = -1)
        self.model.fit(self.X_train, self.y_train)

        # Get the best hyperparameters
        best_params = self.model.best_params_
        print("Best Hyperparameters:", best_params)

        # Use the best model for predictions
        self.model = self.model.best_estimator_
        predictions = self.model.predict(self.X_test)

        # Evaluate the model
        _, _ = self.evaluate(predictions, 'Best Random Forest Model After Hyperparameter Tuning')
        return predictions    

    def tune_baseline_DNN(self, dropout_rate=[0.3, 0.4, 0.5],
                            neurons=[64, 128, 256],
                            learning_rate=[1e-2, 1e-3, 1e-4]):
        
        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()
            
            
        # Wrap the Keras model using KerasClassifier for scikit-learn compatibility
        model = KerasClassifier(build_fn=create_DNN, verbose=0)
        # Define hyperparameters and their ranges for random search
        param_dist = {
            'num_fet': [self.X_train.shape[1]],
            'dropout_rate': dropout_rate,
            'neurons': neurons,
            'learning_rate': learning_rate
        }

        # Perform random search with cross-validation
        self.model = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3 , n_jobs = -1)
        self.model.fit(self.X_train, self.y_train)

        # Get the best hyperparameters and corresponding accuracy
        best_params = self.model.best_params_
        print("Best Hyperparameters:", best_params)

        # Create a new model with the best hyperparameters
        self.model = create_DNN(num_features=self.X_train.shape[1], dropout_rate=best_params['dropout_rate'], neurons=best_params['neurons'], learning_rate=best_params['learning_rate'])
        self.model = compile_DNN(self.model)
        history = train_DNN(self.model)
        predictions = self.model.predict(self.X_test).flatten()
        _, _ = self.evaluate(predictions, 'Best DNN Model After Hyperparameter Tuning')
        return predictions

    def train_logistic(self, class_weight = 'balanced', dual = False, 
                       max_iter = 500, C = 0.1, penalty = 'l2', solver = 'saga'):

        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()
            
            
        self.model = LogisticRegression(class_weight = class_weight, dual = dual, max_iter = max_iter, C = C, penalty = penalty, solver = solver)
        self.model.fit(self.X_train, self.y_train)

        predictions = self.model.predict_proba(self.X_test)[:,1]
        print('class_weight = ', class_weight, ', dual = ', dual, 
                ', max_iter = ', max_iter, ', C = ', C, ', penalty = ', 
                penalty, ', solver = ', solver)
        _, _ = self.evaluate(predictions, 'Logistic Regression')
        return predictions    


    def train_XGB(self,tree_method='gpu_hist', gpu_id=0, 
    scale_pos_weight=89.67005, n_estimators = 800, learning_rate = 0.05, 
                  max_depth = 6, gamma = 0.15, subsample = 0.7):
        
        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()

        self.model = xgb.XGBClassifier(
        tree_method='gpu_hist', gpu_id=0, 
        scale_pos_weight=89.67005,
        n_estimators = n_estimators,
        learning_rate = learning_rate,
        max_depth = max_depth,
        gamma = gamma,
        subsample = subsample
        )
        self.model.fit(self.X_train, self.y_train)

        predictions = self.model.predict_proba(self.X_test)[:,1]
        print('n_estimators = ', n_estimators, ', learning_rate = ', learning_rate, 
                ', max_depth = ', max_depth, ', gamma = ', gamma, ', subsample = ', 
                subsample)
        _, _ = self.evaluate(predictions, 'XGBoost')
        return predictions    

    def train_RF(self, class_weight={0:89.67005,1:10.32995},
                                  n_estimators = 500,
                                  criterion = 'gini',
                                  max_depth = 15, n_jobs= -1, max_features='sqrt'):
        
        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()

        self.model = RandomForestClassifier(class_weight=class_weight,
                                  n_estimators = n_estimators,
                                  criterion = criterion,
                                  max_depth = max_depth, n_jobs= n_jobs)
        self.model.fit(self.X_train, self.y_train)

        predictions = self.model.predict_proba(self.X_test)[:,1]
        print('class_weight = ', class_weight,
                ', n_estimators = ', n_estimators,
                ', criterion = ', criterion,
                ', max_depth = ', max_depth, ', n_jobs = ', n_jobs)
        _, _ = self.evaluate(predictions, 'Random Forest')
        return predictions
    
    def train_DNN(self, dropout_rate=0.4, neurons=64, learning_rate=0.01):
        
        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()
        
        self.model = create_DNN(self.X_train.shape[1], dropout_rate=0.5, neurons=128, learning_rate=0.01)
        self.model = compile_DNN(self.model)
        history = train_DNN(self.model)
        predictions = self.model.predict(self.X_test).flatten()
        print('dropout_rate = ', dropout_rate,
                ', neurons = ', neurons,
                ', learning_rate = ', learning_rate)
        _, _ = self.evaluate(predictions, 'DNN Model')
        return predictions        


    def train_GBM(self, estimator='gbc'):
        
        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()

        train = self.preprocessed[self.preprocessed['month']<6]
        test = self.preprocessed[self.preprocessed['month']>=6]
        
        classifier = setup(data=train, preprocess=False, target='fraud_bool', verbose=0)
        self.model = create_model(estimator)
        
        predictions = self.model.predict_proba(self.X_test)[:,1]
        
        return predictions
    
    def plot_GBM(self, plot):


        plt.figure(figsize=(8, 8))
        plot_model(self.model, plot=plot)

        if plot == 'boundary':

            # Modify the x-axis label
            plt.xlabel('Class 0', fontsize=12)

            # Modify the y-axis label
            plt.ylabel('Class 1', fontsize=12)

        # Show the plot
        plt.show()

    def tune_GBM(self, estimator='gbc', optimize='AUC'):
        
        if self.one_hot == 0:
            self.one_hot_encoder()
        
        if self.std_scale == 0:
            self.standard_scaler()

        self.train_GBM(estimator)
        self.model = tune_model(self.model, optimize = optimize)
        
        predictions = self.model.predict_proba(self.X_test)[:,1]
        
        return predictions
    
    def plot_confusion(self, confusion, model_name):

        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(confusion, cmap = 'YlGnBu',annot = True, fmt='d')
        ax.set_title('Confusion Matrix of '+model_name)
    

    def evaluate_GBM(self):

        pred = self.model.predict(self.X_test)

        accuracy = accuracy_score(self.y_test , pred)
        precision = precision_score(self.y_test , pred)
        recall = recall_score(self.y_test , pred)
        f1 = f1_score(self.y_test,pred)

        print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f},\
        F1: {3:.4f}'.format(accuracy, precision, recall, f1))

        confusion = confusion_matrix(self.y_test, pred)

        self.plot_confusion(confusion, 'GBM')

        fpr, tpr, _ = roc_curve(self.y_test, pred)
        plot_roc(fpr, tpr, 'GBM')

    
