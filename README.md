# Knee decease diagnosis

# AI Application for Knee Disease Diagnosis Using X-ray Images

## Introduction
This research project focuses on applying artificial intelligence (AI) to diagnose knee diseases through X-ray images. The AI model is trained to recognize and classify three main diseases:
1. **Osteoarthritis**
2. **Osteopenia**
3. **Osteoporosis**

## Disease Types and X-ray Characteristics
### 1. Osteoarthritis
Osteoarthritis is a chronic joint disease characterized by cartilage degradation and loss. X-ray indicators include:
- Joint space narrowing
- Subchondral bone sclerosis
- Osteophyte formation

Osteoarthritis severity is assessed based on the **Kellgren-Lawrence grading scale**, which includes five levels:
- **Normal**: No signs of osteoarthritis
- **Doubtful**: Minimal joint space narrowing and possible osteophytes
- **Mild**: Definite osteophytes with mild joint space narrowing
- **Moderate**: Moderate joint space narrowing with multiple osteophytes
- **Severe**: Severe joint space narrowing with subchondral bone sclerosis

### 2. Osteopenia and Osteoporosis
Diagnosis of **osteoporosis** and **osteopenia** is commonly performed using **Dual Energy X-ray Absorptiometry (DEXA)** imaging.

Based on X-ray images, three disease levels are identified:
- **Normal**:
  - Bones have uniform density with no abnormalities.
  - The joint space is even and not narrowed.
  - No osteophytes or other abnormalities.
- **Osteopenia**:
  - Bones appear slightly translucent compared to normal.
  - Bone trabeculae may be more visible due to reduced calcium density.
- **Osteoporosis**:
  - Bones appear significantly translucent.
  - Small fractures or cracks may be visible due to weak bones.
  - Bone trabeculae are prominently visible.

## Models Used
The project evaluates and implements multiple deep learning models, including:
- **Custom CNN**
- **ResNet18**
- **MobileNetV2**
- **LeNet**
- **VGG11**
- **VGG13**
- **VGG16**
- **ViT-B/16 (Vision Transformer)**

## Performance Metrics
The models are assessed based on essential performance metrics:
- **Accuracy**
- **F1 Score**
- **Precision**
- **Recall**
- **ROC Curve**

## Dataset
The dataset is collected from three primary sources:
- [Dataset 1](https://www.dropbox.com/scl/fo/obw0z67349v5huev3pczk/APJ1lnMzem2mEwE2udkjYiQ?rlkey=zc5cme0umcym0cwyeucrpz5d0&st=2nr4iloa&dl=0)
- [Dataset 2](https://data.mendeley.com/datasets/t9ndx37v5h/1)
- [Dataset 3](https://data.mendeley.com/datasets/fxjm8fb6mw/2)

## Result

![](https://s2.ezgif.com/tmp/ezgif-2-bb46c8a4e4.gif)


## Conclusion

This project aims to develop an AI-based system capable of accurately, quickly, and efficiently diagnosing knee diseases from X-ray images. Advanced deep learning models will be implemented to optimize the diagnostic process, improving healthcare quality and assisting medical professionals in decision-making.

