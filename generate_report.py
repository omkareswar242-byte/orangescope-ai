#!/usr/bin/env python3
"""Generate a measured orange-condition classification report. No fabricated benchmarks."""
from pathlib import Path
import json
import argparse
import html
import pandas as pd
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

INK=colors.HexColor('#21343F');ACCENT=colors.HexColor('#B85F17');PALE=colors.HexColor('#FFF1DF');GRAY=colors.HexColor('#536573')

def esc(x):
    return html.escape(str(x))

def pstyles():
    s=getSampleStyleSheet()
    s.add(ParagraphStyle(name='HeadMainOrange',parent=s['Title'],fontName='Helvetica-Bold',fontSize=21,leading=26,spaceAfter=13,textColor=INK))
    s.add(ParagraphStyle(name='SectionOrange',parent=s['Heading2'],fontName='Helvetica-Bold',fontSize=12.1,leading=15,spaceBefore=14,spaceAfter=7,textColor=ACCENT))
    s.add(ParagraphStyle(name='BodyOrange',parent=s['BodyText'],fontSize=9.3,leading=14,spaceAfter=7,textColor=INK))
    s.add(ParagraphStyle(name='SmallOrange',parent=s['BodyText'],fontSize=8.0,leading=11,spaceAfter=5,textColor=GRAY))
    s.add(ParagraphStyle(name='TableOrange',parent=s['BodyText'],fontSize=8.0,leading=10.5,textColor=INK))
    return s

def make_table(rows,widths,s):
    data=[[Paragraph(esc(value),s['TableOrange']) for value in row] for row in rows]
    tab=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),PALE),('LINEBELOW',(0,0),(-1,0),.9,ACCENT),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F8FAFB')]),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
        ('RIGHTPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#E3E6E7')),
    ]))
    return tab

def chart(story,path,max_w=440,max_h=245):
    if Path(path).exists():
        pic=Image(str(path));f=min(max_w/pic.imageWidth,max_h/pic.imageHeight)
        pic.drawWidth*=f;pic.drawHeight*=f;story.extend([pic,Spacer(1,8)])

def create_report(target,summary_file):
    summary_file=Path(summary_file)
    if not summary_file.exists():
        raise FileNotFoundError('Measured summary.json is required; refusing to invent results.')
    data=json.loads(summary_file.read_text(encoding='utf8'))
    if data.get('task')!='orange_visual_condition_classification' or data.get('is_smoke_test'):
        raise ValueError('Refusing to generate a final report for a non-orange or partial-data smoke test.')
    results=summary_file.parent
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    st=pstyles()
    body=lambda x:Paragraph(x,st['BodyOrange'])
    heading=lambda x:Paragraph(x,st['SectionOrange'])
    small=lambda x:Paragraph(x,st['SmallOrange'])
    story=[Paragraph('Orange Visual Condition Classification',st['HeadMainOrange']),
       body('<b>Data Science assignment adaptation | Two-model computer vision comparison</b>'),
       body('<b>Scope clarification:</b> The original Greenpod Labs assignment requested <b>avocado ripening stages</b>. The supplied dataset contains <b>orange appearance/condition labels</b>. At the candidate\'s request, this is an <b>orange-only adaptation</b>, and it must not be represented as completed avocado-ripeness work.'),
       heading('1. Problem definition and actual class labels'),
       body('Input: a photograph labeled as an orange. Output: one of four provided folder classes: <b>blackspot</b>, <b>canker</b>, <b>fresh</b> or <b>grenning</b>. The last name is preserved exactly as spelled in the source. This project classifies a single image; the optional green candidate bounding rectangle in the demo is <b>heuristic localization</b>, not a trained object detector.'),
       heading('2. Dataset inspection, outliers and biases')]
    audit=data['audit']; classes=data['classes']
    story.append(body(f"The uploaded dataset contains <b>{audit['image_count']:,}</b> readable images before exact-duplicate removal. Per-class totals: "+', '.join(f"{esc(c)} ({audit['classes'][c]})" for c in classes)+'.'))
    story.append(body(f"Largest/smallest class ratio: <b>{audit['imbalance_ratio_largest_to_smallest']:.2f}:1</b>. Automated quality screens flagged <b>{audit['dark_count']}</b> dark, <b>{audit['bright_count']}</b> bright, <b>{audit['low_detail_count']}</b> low-detail, <b>{audit['small_count']}</b> small, <b>{audit['exact_duplicate_images']}</b> byte-duplicate instances and <b>{audit['possible_near_duplicate_images']}</b> possible perceptual-hash collisions. These are <b>review flags, not confirmed bad images</b>. Unreadable images: <b>{audit['unreadable_count']}</b>."))
    chart(story,results/'class_balance.png',430,195)
    chart(story,results/'image_quality.png',430,185)
    story.append(body('<b>Observed background bias:</b> a manual contact-sheet check showed several <b>fresh</b> samples with white graphic-like backdrops and attached leaves, while sampled <b>grenning</b> images have dark backgrounds. Source composition, lighting, framing, and lesion severity may therefore correlate with labels. Apparent high accuracy could partly reflect these artifacts rather than biological symptoms. The folder label <b>grenning</b> may be a spelling error; do not silently relabel it. Inspect label semantics and class boundaries with the dataset owner. Multiple photos of the same fruit may produce overoptimistic estimates if a fruit identifier is unavailable.'))
    story.append(heading('3. Data splitting, preprocessing and reproducibility'))
    story.append(body('The original training and test folders are respected. A stratified validation subset is created from the training folder; the provided test folder remains a final untouched evaluation holdout. Exact SHA-256 duplicate copies are removed before model training and split assignment. Images used by the CNN are RGB, uniformly resized to <b>'+str(data['image_size'])+' x '+str(data['image_size'])+'</b>, normalized, and augmented <b>only during training</b> using mild flips, rotations and color jitter.'))
    splits=data['split_counts']; rows=[['Class','Training','Validation','Test']]
    for c in classes: rows.append([c,splits.get('train',{}).get(c,0),splits.get('val',{}).get(c,0),splits.get('test',{}).get(c,0)])
    story.append(make_table(rows,[155,110,110,105],st))
    story.append(small('All splits and image hashes are recorded in results/split_manifest.csv. The original test set is not used for hyperparameter tuning; near-duplicate photographic views may still exist.'))
    story.append(heading('4. Approach A - handcrafted color/texture features + SVM'))
    story.append(body('Color histograms are extracted from HSV and CIELAB channels (24 bins per channel), together with a local binary pattern (LBP) histogram and grayscale texture/contrast statistics. Features are calculated primarily inside a central oval region to reduce - but not eliminate - background effects. A StandardScaler and class-weighted radial-basis-function Support Vector Machine form the classifier.'))
    story.append(body('<b>Why this approach?</b> It is an interpretable, low-data baseline: color and textured surface patterns may help distinguish visible lesion/blemish classes. Weaknesses include fixed hand-crafted features, viewpoint sensitivity, and limited understanding of localized lesions.'))
    story.append(heading('5. Approach B - compact convolutional neural network'))
    story.append(body('A small four-block CNN is trained on RGB images using AdamW and class-weighted cross-entropy. Only training images receive augmentation. The checkpoint with best validation loss is retained, with early stopping after repeated lack of improvement. <b>It was trained from scratch; pretrained weights are not claimed.</b>'))
    story.append(body('<b>Why this approach?</b> A CNN can learn spatial combinations of color, local lesion shape, pattern and texture without explicitly coding each feature. A compact architecture is practical for this dataset and a CPU-only environment, though it may underperform pretrained transfer learning and can overfit on small datasets.'))
    story.append(heading('6. Metrics, model comparison and final selection'))
    story.append(body('<b>Accuracy</b> summarizes correct predictions but can be dominated by frequent classes. <b>Macro precision</b> equally weighs false-positive patterns across four classes, <b>macro recall</b> equally weighs missed classes, and <b>macro F1</b> balances precision and recall across classes. <b>Balanced accuracy</b> averages class recall. Macro F1 is the primary validation selection metric because the orange classes are not perfectly balanced.'))
    rows=[['Model / subset','Accuracy','Macro Precision','Macro Recall','Macro F1','Balanced Acc.']]
    for split_name in ['validation','test']:
        for name,m in data[split_name].items():
            rows.append([name+' / '+('val' if split_name=='validation' else 'test'),
                         f"{m['accuracy']:.3f}",f"{m['macro_precision']:.3f}",f"{m['macro_recall']:.3f}",
                         f"{m['macro_f1']:.3f}",f"{m['balanced_accuracy']:.3f}"])
    story.append(make_table(rows,[126,69,80,76,69,60],st))
    best=data['selected_on_validation_macro_f1']
    testmetrics=data['test'][best]
    story.append(body(f"<b>Model chosen by validation macro F1:</b> {esc(best)}. Its final held-out test accuracy is <b>{testmetrics['accuracy']:.1%}</b>, and test macro F1 is <b>{testmetrics['macro_f1']:.3f}</b>. This is an <b>offline within-dataset evaluation</b>, not production accuracy."))
    chart(story,results/'svm_test_confusion_matrix.png',400,230)
    chart(story,results/(data['cnn_architecture']+'_test_confusion_matrix.png'),400,230)
    for name in ['svm',data['cnn_architecture']]:
        cm_path=results/(name+'_test_confusion_matrix.csv')
        if cm_path.exists():
            cm=pd.read_csv(cm_path,index_col=0)
            diag=cm.to_numpy().diagonal()
            denom=cm.sum(axis=1).to_numpy()
            perclass=', '.join(f'{c}: {100*float(t)/float(n):.1f}%' if n else f'{c}: N/A' for c,t,n in zip(cm.index,diag,denom))
            story.append(small(f'<b>{esc(name)} class-wise test recall:</b> {esc(perclass)}.'))
    story.append(heading('7. One-image inference demonstration'))
    demo=results/'example_prediction.json'
    if demo.exists():
        d=json.loads(demo.read_text(encoding='utf8'))
        story.append(body(f"<b>Held-out example:</b> {esc(Path(d['image']).name)}; actual folder label <b>{esc(d['actual_label_if_known'])}</b>. SVM predicted <b>{esc(d['svm_label'])}</b> (score {d['svm_score']:.3f}); {esc(d['cnn_type'])} predicted <b>{esc(d['cnn_label'])}</b> (softmax score {d['cnn_softmax_score']:.3f}). These scores are <b>uncalibrated</b> and should not be treated as validated probabilities."))
        chart(story,results/'example_prediction.jpg',455,260)
    else: story.append(body('No example file generated. Use predict.py after training.'))
    story.append(heading('8. Mandatory questions and answers'))
    story.append(body('<b>Q1. In production how does your model perform?</b><br/>This system has <b>not been deployed in production</b>. Its measured results are from the held-out orange dataset shown above. In a controlled setting with similar photography, the measured test performance gives an initial estimate, but different lighting, varieties, mobile cameras or previously unseen disease appearances can reduce reliability. Before deployment, benchmark end-to-end CPU/GPU latency on target hardware, evaluate samples from new farms and seasons, monitor per-class recall, and introduce manual review for uncertain cases.'))
    story.append(body('<b>Q2. What is the drawback of your model?</b><br/>The baseline depends on fixed color/texture features; the CNN has limited data and is trained from scratch. Background cues, class ambiguity and lighting may cause false predictions. The system assigns one label per image, so mixed infections or multiple fruit objects cannot be described reliably. The demo rectangle is heuristic, not a trained orange detector. Since fruit/plant IDs are unavailable, multiple photos of a single source may leak similarity across splits.'))
    story.append(body('<b>Q3. If given more time, what would you improve?</b><br/>Audit suspicious samples and source labels with a domain expert; obtain more balanced images from new orchards, lighting conditions and devices; group splits by fruit or acquisition batch; use additional class-sensitive metrics and confidence calibration; compare a pretrained MobileNetV2, experiment with stronger segmentation or object detection for cluttered scenes, and benchmark speed and robustness on representative production hardware.'))
    story.append(body('<b>Q4. Where did you spend more time solving the assignment?</b><br/>The main technical focus was inspecting the supplied dataset, retaining its original holdout split, assessing quality and duplicate risks, establishing a fair comparison between feature-based SVM and CNN, and interpreting per-class errors. The CNN training and evaluation also required repeatable preprocessing and checkpointing. This describes the implemented workflow, not a fabricated personal time log.'))
    story.append(heading('9. Flask demonstration interface'))
    story.append(body('A small <b>Flask</b> web application connects the two saved models to an image-upload page. The user selects a photo, clicks Predict, and sees the SVM and CNN outputs side by side. The page also displays measured test results and confusion matrices from the saved files. This interface demonstrates inference; it does not provide evidence of field-ready performance. Run it locally with <b>python app.py</b> after installing requirements-web.txt.'))
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=52,rightMargin=52,topMargin=50,bottomMargin=50,title='Orange Condition Classification - Two-Approach CV Study')
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(GRAY)
        canvas.drawString(52,30,'Orange condition classification | Tested reproducible pipeline')
        canvas.drawRightString(A4[0]-52,30,f'Page {doc.page}')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return target

if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--summary',default='results/summary.json')
    pa.add_argument('--output',default='report/Orange_Condition_Classification_Report.pdf')
    a=pa.parse_args()
    print(create_report(a.output,a.summary))
