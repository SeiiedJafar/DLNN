import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def confusion_matrix(y_test_class_items, y_pred_class_items):
    tp = np.sum((y_test_class_items == 1) & (y_pred_class_items == 1))  # 1 means alive and 2 means dead
    tn = np.sum((y_test_class_items == 2) & (y_pred_class_items == 2))
    fp = np.sum((y_test_class_items == 2) & (y_pred_class_items == 1))
    fn = np.sum((y_test_class_items == 1) & (y_pred_class_items == 2))
    return tp, tn, fp, fn


def metrics(tp, tn, fp, fn):
    total = tn + tp + fn + fp
    positives = tp + fn
    negatives = tn + fp
    sensitivity = tp / positives
    specificity = tn / negatives
    precision = tp / (tp + fp)
    return total, positives, negatives, sensitivity, specificity, precision


df = pd.read_csv('cleaned_data.csv')
result_df = pd.DataFrame()

targetValue = 'Overall Survival Status'
ignoredValues = [targetValue, "Patient's Vital Status", "Patient ID"]

scaler = StandardScaler()
features = scaler.fit_transform(df.drop(ignoredValues, axis=1).values)
target = df[targetValue].values
# target = scaler.fit_transform(target.reshape(-1, 1)).flatten()

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

patientsMissedVitalStatus = x_test[y_test == 0]

LR_model = LinearRegression()
LR_model.fit(x_train, y_train)
y_pred = LR_model.predict(x_test).astype('float64')
y_pred = np.round(y_pred).astype('int64')
result_df = result_df.assign(test=y_test, LR_pred=y_pred)

DT_model = tree.DecisionTreeRegressor(min_samples_leaf=3, random_state=42)
DT_model.fit(x_train, y_train)
y_pred = DT_model.predict(x_test).astype('float64')
y_pred = np.round(y_pred).astype('int64')
result_df = result_df.assign(DT_pred=y_pred)

RF_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
RF_model.fit(x_train, y_train)
y_pred = RF_model.predict(x_test).astype('float64')
y_pred = np.round(y_pred).astype('int64')
result_df = result_df.assign(RF_pred=y_pred)

print(f'Result before removing unknown status in test: {result_df.shape[0]}')
result_df = result_df[result_df['test'] != 0]
print(f'Result after removing unknown status in test: {result_df.shape[0]},\n')
print(result_df.sample(10))

TP, TN, FP, FN = confusion_matrix(result_df['LR_pred'], result_df['test'])
total, positives, negatives, sensitivity, specificity, precision = metrics(TP, TN, FP, FN)
print(f'LR Confusion Matrix : TN :{TN} , FP : {FP} , TP : {TP} , FN : {FN}')
print(f'Total : {total} , positives : {positives} , negatives : {negatives}')
print(f'Precision : {precision} , Recall or sensitivity : {sensitivity} , specificity : {specificity},\n')

TP, TN, FP, FN = confusion_matrix(result_df['DT_pred'], result_df['test'])
total, positives, negatives, sensitivity, specificity, precision = metrics(TP, TN, FP, FN)
print(f'DT Confusion Matrix : TN :{TN} , FP : {FP} , TP : {TP} , FN : {FN}')
print(f'Total : {total} , positives : {positives} , negatives : {negatives}')
print(f'Precision : {precision} , Recall or sensitivity : {sensitivity} , specificity : {specificity},\n')

TP, TN, FP, FN = confusion_matrix(result_df['RF_pred'], result_df['test'])
total, positives, negatives, sensitivity, specificity, precision = metrics(TP, TN, FP, FN)
print(f'RF Confusion Matrix : TN :{TN} , FP : {FP} , TP : {TP} , FN : {FN}')
print(f'Total : {total} , positives : {positives} , negatives : {negatives}')
print(f'Precision : {precision} , Recall or sensitivity : {sensitivity} , specificity : {specificity},\n')
