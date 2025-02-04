# Knee decease diagnosis

# AI Application for Knee Disease Diagnosis Using X-ray Images

## Introduction
This research project focuses on applying artificial intelligence (AI) to diagnose knee diseases through X-ray images. The AI model is trained to recognize and classify three main diseases:
1. **Osteoarthritis**
2. **Osteopenia**
3. **Osteoporosis**

## Disease Types and X-ray Characteristics

**Normal knee without any trouble:**
  - Bones have uniform density with no abnormalities.
  - The joint space is even and not narrowed.
  - No osteophytes or other abnormalities.

![normal](https://github.com/user-attachments/assets/c245e436-362c-4823-9a6c-555189bd11b6)


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

**The following  images (from left to right) show severities of the decease. The last three images reveal the spikes growing from bones.**

![Osteoarthritis](https://github.com/user-attachments/assets/efef158c-5102-4f7a-8de9-095185389e57)

Comparing **normal** and **Severe**: Space narrowing significantly

![Osteoarthritis](https://github.com/user-attachments/assets/226d0e47-4956-4327-bdda-cec4168340ac)

### 2. Osteopenia and Osteoporosis

Diagnosis of **osteoporosis** and **osteopenia** is commonly performed using **Dual Energy X-ray Absorptiometry (DEXA)** imaging.

- **Osteopenia**:
  - Bones appear slightly translucent compared to normal.
  - Bone trabeculae may be more visible due to reduced calcium density.
- **Osteoporosis**:
  - Bones appear significantly translucent.
  - Small fractures or cracks may be visible due to weak bones.
  - Bone trabeculae are prominently visible.

![Picture1](https://github.com/user-attachments/assets/6a3a5696-56d5-4a95-b8ea-e5a798e7d0c9)


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

Dataset's format:
```
├───train
│   ├───normal
│   ├───oa_doubtful
│   ├───oa_mild
│   ├───oa_moderate
│   ├───oa_severe
│   ├───osteopenia
│   └───osteoporosis
└───val
    ├───normal
    ├───oa_doubtful
    ├───oa_mild
    ├───oa_moderate
    ├───oa_severe
    ├───osteopenia
    └───osteoporosis
```
Each of `[normal, oa_doubtful, oa_mild, oa_moderate, oa_severe, osteopenia, osteoporosis]` is label of each image.

## Result


![output](https://github.com/user-attachments/assets/41aa42db-a263-4c67-9d33-71219df84616)


## Conclusion

This project aims to develop an AI-based system capable of accurately, quickly, and efficiently diagnosing knee diseases from X-ray images. Advanced deep learning models will be implemented to optimize the diagnostic process, improving healthcare quality and assisting medical professionals in decision-making.

