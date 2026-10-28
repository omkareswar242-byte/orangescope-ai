#!/usr/bin/env python3
"""Classify one image with BOTH trained approaches and annotate a candidate fruit ROI."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from orange_cv.data import read_image
from orange_cv.roi import find_fruit_bbox
from orange_cv import classical,cnn

def predict_one(image_path,models_dir='models',results_dir='results',expected=None):
    md=Path(models_dir);rd=Path(results_dir);rd.mkdir(parents=True,exist_ok=True)
    classes=json.loads((md/'classes.json').read_text(encoding='utf-8'))
    info=__import__('torch').load(md/'cnn_best.pt',map_location='cpu',weights_only=True)
    architecture=info['architecture']; size=info['size']
    df=pd.DataFrame({'path':[str(Path(image_path).resolve())],'label':[classes[0]]})
    svm=classical.load_model(md/'classical_svm.joblib')
    smodel,ps=classical.predict(svm,df)
    dmodel=cnn.load_model(md/'cnn_best.pt')
    dclasses=list(info['classes'])
    dpred,pd_probs=cnn.predict(dmodel,df,dclasses,architecture,size)
    image=read_image(image_path)
    box=find_fruit_bbox(image)
    shown=image.copy()
    if box is not None:
        x,y,w,h=box
        cv2.rectangle(shown,(x,y),(x+w,y+h),(10,180,20),3)
    message=f'SVM: {smodel[0]} ({ps.max():.2f}) | {architecture}: {dpred[0]} ({pd_probs.max():.2f})'
    cv2.rectangle(shown,(0,0),(shown.shape[1],43),(255,255,255),-1)
    cv2.putText(shown,message[:85],(8,28),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,55,20),2,cv2.LINE_AA)
    output=rd/'example_prediction.jpg'
    cv2.imwrite(str(output),shown)
    result={'image':str(Path(image_path).resolve()),'actual_label_if_known':expected,
            'svm_label':smodel[0],'svm_score':round(float(ps.max()),4),
            'cnn_type':architecture,'cnn_label':dpred[0],'cnn_softmax_score':round(float(pd_probs.max()),4),
            'candidate_roi_bbox_xywh':list(box) if box else None,'annotated_image':str(output),
            'warning':'Scores are model confidence estimates, not calibrated probabilities. Candidate ROI is heuristic.'}
    (rd/'example_prediction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True)
    p.add_argument('--models-dir',default='models')
    p.add_argument('--results-dir',default='results')
    p.add_argument('--expected',default=None,help='Optional known reference label; never inferred from the image')
    a=p.parse_args()
    print(json.dumps(predict_one(a.image,a.models_dir,a.results_dir,a.expected),indent=2))
