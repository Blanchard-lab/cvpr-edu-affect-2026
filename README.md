# Evaluating Web-trained Facial Expression Recognition Models in Collaborative Learning
<img width="649" height="451" alt="teaser (1)" src="https://github.com/user-attachments/assets/75b37e15-346d-4633-ada7-044cc2be4312" />



This repository contains the experimental pipeline used in our study evaluating how pretrained facial expression recognition (FER) models behave in collaborative learning settings. The goal of this project is to examine whether common FER outputs—categorical basic emotions and dimensional valence–arousal representations—align with epistemic affective states such as curiosity, confusion, and frustration observed in collaborative problem-solving.

The repository provides tools for dataset preprocessing, running inference using multiple FER models, and performing analyses including cross-taxonomy alignment, dimensional affect structure, and cross-model agreement.

---

## Overview

Most FER models are trained on large-scale web datasets such as AffectNet. These datasets are annotated using basic emotion categories (e.g., happy, sad, anger) and sometimes dimensional representations (valence and arousal).

However, affective states relevant to learning are often **epistemic emotions**, including:

- Curious
- Confused
- Frustrated
- Optimistic
- Disengaged
- Surprised
- Conflicted

This project evaluates whether pretrained FER models produce meaningful signals for these states when applied to collaborative learning data.

The experimental workflow includes:

1. Extracting and sampling face crops around affect reports
2. Running pretrained FER models
3. Aggregating predictions per affect instance
4. Evaluating alignment between model outputs and epistemic labels
5. Comparing agreement across different FER architectures
6. Comparing behavior on educational data vs AffectNet benchmark data

---


---

## Pipeline

### 1. Frame Sampling

For each affect report at time \(t\), we collect all cropped face images within a ±5 second window:

From this pool we randomly sample **K = 10 frames** to represent the instance.

This creates a dataset of labeled face crops used for model inference.

---

### 2. Model Inference

We evaluate several pretrained FER models.

#### Categorical emotion models

- OpenFace 3.0
- LibreFace
- POSTER++
- DDAMFN

Each frame produces a categorical emotion prediction.

Instance-level prediction is obtained via:

- **OpenFace:** mean probability vector
- **Other models:** majority vote across sampled frames

---

#### Dimensional models

- HSEmotion

Instance-level predictions are computed by averaging across frames.

---

### 3. Evaluation

We perform three main analyses.

#### Cross-taxonomy alignment

Compare predicted basic emotions with epistemic labels using confusion matrices.

Goal: determine whether epistemic states map to consistent emotion predictions.

---

#### Dimensional affect structure

Visualize valence–arousal outputs:

- scatter plots
- box plots
- clustering behavior

Goal: determine whether epistemic states occupy distinct regions in affect space.

---

#### Cross-model agreement

Compare predictions between different FER models.

Example:
OpenFace vs LibreFace


Low agreement indicates instability under domain shift.

---

### 4. AffectNet Control Experiment

To isolate domain shift effects, we run the same models on a subset of AffectNet.

Procedure:

1. Randomly sample N images from AffectNet
2. Run OpenFace and LibreFace
3. Compute cross-model confusion matrix


**TO REPRODUCE OUR EXPERIMENTS SIMPLY RUN THE SCRIPTS IN THE ORDER**

To modify/extend, consult src/ and write additional scripts to apply your added functionality in src. 
