import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree

def confusion_matrix(y_test_class_items, y_pred_class_items):
    tp = np.sum((y_test_class_items == 1) & (y_pred_class_items == 1))
    tn = np.sum((y_test_class_items == 0) & (y_pred_class_items == 0))
    fp = np.sum((y_test_class_items == 0) & (y_pred_class_items == 1))
    fn = np.sum((y_test_class_items == 1) & (y_pred_class_items == 0))
    return tp, tn, fp, fn
def metrics(TN, TP, FN, FP):
    total = TN + TP + FN + FP
    positives = TP + FN
    negatives = TN + FP
    sensitivity = TP / positives
    specificity = TN / negatives
    precision = TP / (TP + FP)
    return total, positives, negatives, sensitivity, specificity, precision
def roc_curve_scratch(y_test, y_predict):
    thresholds = np.linspace(np.min(target), np.max(target), 2000)
    roc_fp_rate, roc_rp_rate = [], []
    auc_value = 0.0
    for item in thresholds:
        y_pred = (y_predict >= item).astype(int)
        thresholds_tp, thresholds_tn, thresholds_fp, thresholds_fn = confusion_matrix(y_test, y_pred)
        tpr_value = thresholds_tp / (thresholds_tp + thresholds_fn)
        fpr_value = thresholds_fp / (thresholds_fp + thresholds_tn)
        roc_rp_rate.append(tpr_value)
        roc_fp_rate.append(fpr_value)

    for i in range(1, len(thresholds)):
        y_pred_high = (y_predict >= thresholds[i]).astype(int)
        y_pred_low = (y_predict >= thresholds[i - 1]).astype(int)
        tp_high, tn_high, fp_high, fn_high = confusion_matrix(y_test, y_pred_high)
        tp_low, tn_low, fp_low, fn_low = confusion_matrix(y_test, y_pred_low)
        tpr_high = tp_high / (tp_high + fn_high)
        fpr_high = fp_high / (fp_high + tn_high)
        tpr_low = tp_low / (tp_low + fn_low)
        fpr_low = fp_low / (fp_low + tn_low)
        auc_value += 0.5 * (tpr_high + tpr_low) * (fpr_low - fpr_high)

    roc_fp_rate[0] = 1
    roc_fp_rate[-1] = 0
    roc_rp_rate[0] = 1
    roc_rp_rate[-1] = 0
    return roc_fp_rate, roc_rp_rate, thresholds, auc_value
def roc_curve_graph(roc_fpr, roc_tpr, auc_value, save_path='roc_curves.jpg'):
    plt.figure(figsize=(10, 10))
    plt.plot(roc_fpr, roc_tpr, label=f'AUC = {auc_value:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve (Custom Implementation)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

input_df = pd.read_csv('data.csv')
features = input_df.drop(['J1', 'J2', 'J3'], axis=1).values
target = input_df['J3'].values
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

DT_model = tree.DecisionTreeRegressor(max_depth=20, random_state=42)
DT_model.fit(x_train, y_train)
y_pred = DT_model.predict(x_test)

pred_df = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})
pred_df['y_pred_class'] = (pred_df['y_pred'] > 22.5).astype(int)
pred_df['y_test_class'] = (pred_df['y_test'] > 22.5).astype(int)

TP, TN, FP, FN = confusion_matrix(pred_df['y_test_class'].values, pred_df['y_pred_class'].values)
total, positives, negatives, sensitivity, specificity, precision = metrics(TN, TP, FN, FP)
roc_fpr, roc_tpr, thresholds, auc_value = roc_curve_scratch(pred_df['y_test_class'], y_pred)
print(f'Total : {total} , positives : {positives} , negatives : {negatives}\nTP : {TP} , FN : {FN} , TN :{TN} , FP : {FP}\nPrecision : {precision:.5f} , Recall(sensitivity) : {sensitivity:.5f} , specificity : {specificity:.5f}')

roc_curve_graph(roc_fpr, roc_tpr, auc_value)
