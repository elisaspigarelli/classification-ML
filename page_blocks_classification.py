#########################################################################

# Cognome : Spigarelli
# Nome : Elisa
# Dataset : Page Blocks

#########################################################################
import warnings

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
import sklearn.linear_model as lm
from sklearn.metrics import  f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.simplefilter(action='ignore')

##########################################################################
#                       TOOL FUNCTION                                    #
##########################################################################

def distplot(feature, frame, color='g'):
    plt.figure(figsize=(8, 3))
    plt.title("Distribuzione di {}".format(feature))
    ax = sns.distplot(frame[feature], color=color)
    plt.close()

def plot_confusion_matrix(conf_matrix):
    plot_conf = sns.heatmap(pd.DataFrame(conf_matrix, index=labels, columns=labels),
                            square=True, cbar=False,
                            cmap='RdBu_r',
                            xticklabels=labels,
                            yticklabels=labels,
                            annot=True)
    plot_conf.set_xticklabels(plot_conf.get_xticklabels(),
                              rotation=20)
    bottom, top = plot_conf.get_ylim()
    plot_conf.set_ylim(bottom + 0.5, top - 0.5)


def plot_classification_report(y_test, y_pred):
    models_report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    ax=sns.heatmap(pd.DataFrame(models_report).iloc[:-1, :].T, annot=True, vmin=0, vmax=1)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

###########################################################################
#                                   EDA                                    #
###########################################################################

# load Dataset
dataset_filename = 'page_blocks_data/page-blocks.data'

df = pd.read_csv(dataset_filename, sep='\s+',
                 names=['height', 'length', 'area', 'eccen', 'p_black', 'p_and', 'mean_tr', 'blackpix', 'blackand', 'wb_trans', 'Class'])

labels=["(1)text ", "(2)horiz. line ", "(3)graphic", "(4)vert. line", "(5)picture"]

# check dataset
df.head()
df.info()
df.describe()

# build design matrix and target vector
y = df.Class.values  #df['Class']
X = df.drop(columns='Class')

#visualized points distributions in class

plt.figure()
ax = sns.countplot(y="Class", data=df)
total = len(df['Class'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        cx = p.get_x() + p.get_width() + 2
        cy = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (cx, cy))
bottom, top = ax.get_xlim()
ax.set_xlim(bottom, top + 250)
plt.title('Frequenza delle classi')
plt.xlabel('count')
plt.show()

# distribution for features
columns = ['height', 'length', 'area', 'eccen', 'p_black', 'p_and', 'mean_tr', 'blackpix', 'blackand', 'wb_trans']

# show features' distribution
for feat in columns:
    distplot(feat, X)

plt.figure()
X.boxplot()
plt.title("Rappresentazione della scala dei valori")
plt.show()
plt.close()

##################################################################################
#                                  PREPROCESSING                                 #
##################################################################################

# Correlation matrix for collinear features
correlation_matrix = X.corr()
fig = plt.figure(figsize=(10, 6))
ax = sns.heatmap(correlation_matrix,
                 square=True,
                 vmin=-1, vmax=1, center=0,
                 cmap="YlGnBu",
                 xticklabels=correlation_matrix.columns,
                 yticklabels=correlation_matrix.columns, annot=True,
                 )
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Matrice di Correlazione")
plt.tight_layout()
plt.show()
plt.close()

# Feature Importance
params = {'random_state': 0, 'n_jobs': 4, 'n_estimators': 5000, 'max_depth': 8}
clf = RandomForestClassifier(**params)
clf = clf.fit(X, y)

imp = pd.Series(data=clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure()
plt.title("Importanza delle features")
ax = sns.barplot(y=imp.index, x=imp.values, palette="Blues_d", orient='h')
plt.close()

#features selection
X = X.drop(columns='blackand')
columns = ['height', 'length', 'area', 'eccen', 'p_black', 'p_and', 'mean_tr', 'blackpix',  'wb_trans']
########################################################################
#                           SPLIT DATASET                              #
########################################################################

# split dataset in training-set e test-set keeping the initial stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# plot class after split in stratify=y

data_train=pd.concat([X_train, pd.DataFrame(y_train, columns=['Class'])],axis=1, sort=False)
plt.figure()
ax = sns.countplot(y="Class", data=data_train)
total = len(y_train)
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        cx = p.get_x() + p.get_width() + 2
        cy = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (cx, cy))
bottom, top = ax.get_xlim()
ax.set_xlim(bottom, top + 250)
plt.title('Distribuzione delle classi sul Training set')
plt.xlabel('count')
plt.show()

data_test=pd.concat([X_test, pd.DataFrame(y_test, columns=['Class'])],axis=1, sort=False)
plt.figure()
ax = sns.countplot(y="Class", data=data_test)
total = len(y_test)
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        cx = p.get_x() + p.get_width() + 2
        cy = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (cx, cy))
bottom, top = ax.get_xlim()
ax.set_xlim(bottom, top + 50)
plt.title('Distribuzioni delle classi sul Test set')
plt.xlabel('count')
plt.show()

# Standardization of X_train and X_test
std = StandardScaler()
X_train_std = pd.DataFrame(std.fit_transform(X_train), columns=columns)
X_test_std = pd.DataFrame(std.transform(X_test), columns=columns)

plt.figure()
X_train_std.boxplot()
plt.title("Rappresentazione della scala dei valori standardizzati")
plt.show()
plt.close()

# show features' distribution after standardization
#for feat in columns:
#    distplot(feat, X_train_std, 'b')

# overlook at variables with standardization
dataset= pd.concat([X_train_std, pd.DataFrame(y_train, columns=['Class'])],axis=1, sort=False)
sns.set(style="ticks", color_codes=True)
g = sns.pairplot(dataset, hue="Class")
plt.show()
plt.close()
########################################################################
#                             MODELS                                   #
########################################################################

models_name = []

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(weights='distance',  algorithm='auto', p=2, metric='minkowski')
models_name.append('K-Nearest Neighbors')

# Softmax Regression
logr_model = lm.LogisticRegression(penalty='l2', multi_class='multinomial', solver='newton-cg')
models_name.append('Softmax Regression')

# Support Machine Vector
svc_model = SVC(kernel='rbf', class_weight='balanced', decision_function_shape='ovo')
models_name.append('Support Vector Machine')

# MLP
mlp_model = MLPClassifier(hidden_layer_sizes=(400,), max_iter=500, verbose=False, early_stopping=True, validation_fraction=0.5)
models_name.append('MLP')

# Naive bayes
Naive_model = GaussianNB()
models_name.append('Naive Bayes')

###########################################################################
#                  CROSS - VALIDATION FOR MODEL SELECTION                 #
###########################################################################

# KNN MODEL SELECTION

parameters = {'n_neighbors': range(1, 10, 2)}
grid = GridSearchCV(estimator=knn_model, param_grid=parameters, scoring='f1_weighted', cv=10)
grid.fit(X_train_std, y_train)
print('K-Nearest Neighbor')
print('the best value for parameter k is ', grid.best_params_.get('n_neighbors'),
      '\nsince it leads to F1-score = ', grid.best_score_)
knn_model = grid.best_estimator_

# SOFTMAX MODEL SELECTION

parameters = {'C': [ 0.5, 0.7, 1, 1.3, 1.5], 'class_weight': [None, 'balanced']}
grid = GridSearchCV(estimator=logr_model, param_grid=parameters, scoring='f1_weighted', cv=10)
grid.fit(X_train_std, y_train)
print('Softmax Regression')
print('the best value for parameter C is ', grid.best_params_.get('C'),
      '\nthe best choice for parameter class_weight is ', grid.best_params_.get('class_weight'),
      '\nsince these lead to F1-score = ', grid.best_score_)
logr_model = grid.best_estimator_

# SVC MODEL SELECTION

parameters = {'C': [1, 3, 5, 10, 50, 100], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01]}
grid = GridSearchCV(estimator=svc_model, param_grid=parameters, scoring='f1_weighted', cv=10)
grid.fit(X_train_std, y_train)
print("SVC")
print('the best value for parameter C is ', grid.best_params_.get('C'),
      '\nthe best choice for parameter gamma is', grid.best_params_.get('gamma'),
      '\nsince the lead to F1_score = ', grid.best_score_)
svc_model = grid.best_estimator_

# MLP MODEL SELECTION

parameters = {'activation': ['logistic', 'tanh'],
              'solver': ['adam', 'sgd']}
grid = GridSearchCV(estimator=mlp_model, param_grid=parameters, scoring='f1_weighted', cv=10)
grid.fit(X_train_std, y_train)
print(" MLP ")
print('the best choice for parameter activation is ', grid.best_params_.get('activation'),
      '\nthe best choice for parameter solver is ', grid.best_params_.get('solver'),
      '\nsince the lead to F1-score = ', grid.best_score_)
mlp_model = grid.best_estimator_

# NAIVE BAYES MODEL SELECTION
# The GaussianNB model hasn't a group of tuning parameters to estimate from cross-validation


# Now it's show how works cross-validation using validation curve (only for two model)

# Validation Curve with KNN
K=range(1,10,1)
knn2_model = KNeighborsClassifier(weights='distance',  algorithm='auto', p=2, metric='minkowski')
train_scores, test_scores = validation_curve(knn2_model, X_train_std, y_train, param_name= "n_neighbors", param_range= K, scoring="f1_weighted", cv=10, n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Validation Curve con K-Nearest Neighbors")
plt.xlabel("k")
plt.ylabel("Score")
plt.ylim(0.7, 1.1)
lw = 2
plt.plot(K, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(K, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(K, test_scores_mean, label="Cross-validation score",
             color="navy", marker='o', lw=lw)
plt.fill_between(K, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

# Validation curve with Logistic Regression
C=[0.5, 0.7, 1, 1.3, 1.5]
logr2_model = lm.LogisticRegression(class_weight=None, penalty='l2', multi_class='multinomial', solver='newton-cg')
train_scores, test_scores = validation_curve(logr2_model, X_train_std, y_train, param_name= "C", param_range= C, scoring="f1_weighted", cv=10, n_jobs=4)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Validation Curve con Logistic Regression")
plt.xlabel("C")
plt.ylabel("Score")
plt.ylim(0.7, 1.1)
lw = 2
plt.plot(C, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(C, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(C, test_scores_mean, label="Cross-validation score",
             color="navy", marker='o', lw=lw)
plt.fill_between(C, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

###########################################################################
#                  CROSS - VALIDATION FOR MODEL EVALUATION                #
###########################################################################

# an example of the expected value from the prediction on the cross-validation test fold
scores = cross_validate(knn_model, X_train_std, y_train, cv=10, scoring=('f1_weighted', 'balanced_accuracy'))
print('The cross-validated F1-score of your algorithm is ', np.mean(scores['test_f1_weighted']))
print('The cross-validated Balanced Accuracy of your algorithm is ', np.mean(scores['test_balanced_accuracy']))

############################################################################
#                         TRAIN                                            #
############################################################################

# Train model(s)
knn_model.fit(X_train_std, y_train)
logr_model.fit(X_train_std, y_train)
svc_model.fit(X_train_std, y_train)
mlp_model.fit(X_train_std, y_train)
Naive_model.fit(X_train_std, y_train)

############################################################################
#                             PREDICTION                                   #
############################################################################

y_predicted = []

# Test model(s)
y_pred_knn_model = knn_model.predict(X_test_std)
y_predicted.append(y_pred_knn_model)

y_pred_logr_model = logr_model.predict(X_test_std)
y_predicted.append(y_pred_logr_model)

y_pred_svc = svc_model.predict(X_test_std)
y_predicted.append(y_pred_svc)

y_pred_MLP = mlp_model.predict(X_test_std)
y_predicted.append(y_pred_MLP)

y_pred_gnb = Naive_model.predict(X_test_std)
y_predicted.append(y_pred_gnb)

########################################################################
#                           EVALUATION                                 #
########################################################################

# plot confusion matrix and the respective classification report
for i in range(len(models_name)):
    # subplot the two graph for each model
    plt.figure()
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(confusion_matrix(y_test, y_predicted[i]))
    plt.subplot(1, 2, 2)
    plot_classification_report(y_test, y_predicted[i])
    plt.suptitle("Confusion Matrix and Classification Report of : %s"% models_name[i])
    plt.show()

    print('RESULTS OF THE %s CLASSIFIER' % models_name[i])
    print('Accuracy: ', balanced_accuracy_score(y_test, y_predicted[i]))
    print('Precision: ', precision_score(y_test, y_predicted[i], average='weighted'))
    print('Recall: ', recall_score(y_test, y_predicted[i], average='weighted'))
    print('F1-Score: ', f1_score(y_test, y_predicted[i], average='weighted'))


