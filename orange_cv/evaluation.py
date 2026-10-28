"""Held-out evaluation, diagnostic plots and fair macro-F1 comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix


def evaluate(name, split_name, truth, pred, classes, results_dir):
    out=Path(results_dir);out.mkdir(parents=True,exist_ok=True)
    precision,recall,f1,_=precision_recall_fscore_support(truth,pred,labels=classes,average='macro',zero_division=0)
    weighted=precision_recall_fscore_support(truth,pred,labels=classes,average='weighted',zero_division=0)[2]
    metrics={'model':name,'split':split_name,'samples':len(truth),
             'accuracy':float(accuracy_score(truth,pred)),
             'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
             'macro_precision':float(precision),'macro_recall':float(recall),
             'macro_f1':float(f1),'weighted_f1':float(weighted)}
    prefix=name+'_'+split_name
    (out/(prefix+'_classification_report.txt')).write_text(classification_report(truth,pred,labels=classes,zero_division=0),encoding='utf8')
    cm=confusion_matrix(truth,pred,labels=classes)
    pd.DataFrame(cm,index=classes,columns=classes).to_csv(out/(prefix+'_confusion_matrix.csv'))
    fig,ax=plt.subplots(figsize=(max(6,1.3*len(classes)),max(5,1.1*len(classes))))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Greens',xticklabels=classes,yticklabels=classes,ax=ax)
    ax.set(title=f'{name} | {split_name} confusion matrix',xlabel='Predicted class',ylabel='True class')
    fig.tight_layout();fig.savefig(out/(prefix+'_confusion_matrix.png'),dpi=150);plt.close(fig)
    return metrics


def plot_dataset(stats,out):
    out=Path(out)
    fig,ax=plt.subplots(figsize=(8,4))
    stats.label.value_counts().sort_index().plot(kind='bar',ax=ax,color='#2d8659')
    ax.set(title='Number of images by actual dataset class',xlabel='Class',ylabel='Images')
    fig.tight_layout();fig.savefig(out/'class_balance.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].hist(stats.mean_brightness,bins=24,color='#4e7c69');axes[0].set(title='Image brightness',xlabel='Mean grayscale intensity')
    axes[1].hist(stats.blur_laplacian,bins=24,color='#6b83a0');axes[1].set(title='Laplacian detail score',xlabel='Variance of Laplacian')
    fig.tight_layout();fig.savefig(out/'image_quality.png',dpi=150);plt.close(fig)
