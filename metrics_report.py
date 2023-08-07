import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, recall_score, precision_score, f1_score, auc, precision_recall_curve

def cm(y_true, y_pred, output_dir, model_name, ts, save=True):

    cm = confusion_matrix(y_true, y_pred,normalize=None)
    cm_df = pd.DataFrame(cm, index=['SITB-', 'SITB+'], columns=['SITB-', 'SITB+'])
    cm_df_meaning = pd.DataFrame([['TN', 'FP'],['FN','TP']], index=['SITB-', 'SITB+'], columns=['SITB-', 'SITB+'])

    cm_norm = confusion_matrix(y_true, y_pred,normalize='all')
    cm_norm = (cm_norm*100).round(2)
    cm_df_norm = pd.DataFrame(cm_norm, index=['SITB-', 'SITB+'], columns=['SITB-', 'SITB+'])
    

    plt.rcParams['figure.figsize'] = [4,4]
    cm_display = ConfusionMatrixDisplay(cm_norm,display_labels=['SITB-', 'SITB+']).plot()
    # todo save

    if save:
        cm_df_meaning.to_csv(output_dir+f'cm_meaning_{model_name}_{ts}.csv')
        cm_df.to_csv(output_dir+f'cm_{model_name}_{ts}.csv')
        cm_df_norm.to_csv(output_dir+f'cm_norm_{model_name}_{ts}.csv')
        


    return cm_df_meaning, cm_df, cm_df_norm


    

def classification_report(y_true, y_pred,y_pred_proba_1, output_dir, model_name, ts):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    np.set_printoptions(suppress=True)
    roc_auc = roc_auc_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    
    # calculate precision and recall for each threshold
    lr_precision, lr_recall, thresholds = precision_recall_curve(y_true, y_pred_proba_1)

    # TODO: add best threshold
    fscore = (2 * lr_precision * lr_recall) / (lr_precision + lr_recall)
    fscore[np.isnan(fscore)] = 0
    ix = np.argmax(fscore)
    best_threshold = thresholds[ix].item()


    pr_auc = auc(lr_recall, lr_precision)
    # AU P-R curve is also approximated by avg. precision
    # avg_pr = metrics.average_precision_score(y_true,y_pred_proba_1)

    sensitivity = recall_score(y_true,y_pred)
    specificity = tn / (tn+fp) # OR: recall_score(y_true,y_pred, pos_label=0)
    precision = precision_score(y_true,y_pred)

    results = pd.DataFrame([sensitivity, specificity,precision,f1, roc_auc,pr_auc, best_threshold], 
                        index = ['Sensitivity', 'Specificity', 'Precision', 'F1', 'ROC AUC','PR AUC', 'Best th PR AUC']).T.round(2)

    results.to_csv(output_dir+f'results_{model_name}_{ts}.csv')
    return results

