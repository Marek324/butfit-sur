# SUR 2025/2026 Project Assignment

**Points:** 25 points

## Task

The task is to train a detector of a single person from a face image and a voice recording. Training samples are available in an archive at:

<https://www.fit.vutbr.cz/study/courses/SUR/public/projekt_2025-2026/SUR_projekt2025-2026.zip>

## Data Structure

The archive contains the following directories:

- `target_train`
- `target_dev`

These contain training samples of the target (detected) person in **PNG** and **WAV** formats.

The directories:

- `non_target_train`
- `non_target_dev`

contain negative examples (non-target samples).

## Data Restrictions and Augmentation

For training the detectors, you are only allowed to use the **provided data**. It is **NOT PERMITTED** to use any other external speech or image data, nor any pre-trained models (e.g. for extracting face or voice embeddings).

However, you may *augment* the data by creating new training samples, for example by:

- adding noise to recordings or images,
- rotating, shifting, or resizing images,
- changing speech speed, etc.

## Generalization

An important goal of the project is the system's ability to **generalize** to unseen data even with the very limited amount of provided training data. The evaluation data (published later, see below) will contain samples with different properties (e.g. noise, changes in image quality, etc.) that may not exactly match the training data.

When designing the system, consider:

- suitable data augmentation techniques,
- model choice with respect to the limited amount of data,
- prevention of overfitting.

These aspects must be clearly described in the documentation (see below).

## Baseline Systems

During the project assignment lecture, simple baseline systems will be demonstrated (**MFCC + GMM** for audio, **linear classifier** for images). To achieve the full score, it is **not sufficient to simply reproduce or slightly modify** these systems. Your own approach reflecting the above requirements is expected.

## Validation Strategy

The data was split into `*_train` and `*_dev` directories mainly for training and testing the baseline systems. Due to the limited amount of data, such a split will not be optimal, and achieving the full score will require an appropriate validation strategy (e.g. **cross-validation** enabling efficient training and testing on all data).

For this purpose, it may be useful to respect information about which training samples belong to the same person and whether they were recorded in the same recording session. The filename of each sample is divided into fields by underscores:

```
f401_01_f21_i0_0.png
 │    │
 │    └── recording session number (01)
 └── person identity (f401)
```

We recommend using cross-validation that respects the data structure (e.g. by recording sessions, so that samples from one session are not simultaneously in both the training and validation sets). The chosen evaluation method used during development must be described and justified in the documentation.

## Evaluation

The final evaluation data, on which your systems will be scored, will be available on **Sunday, May 3rd in the morning**. This data will contain on the order of hundreds of files to process. Your task will be to automatically process this data with your systems (we trust that you will not cheat by looking at the images or listening to the recordings) and upload the result files to the Information System (IS).

### Result File Format

ASCII file with three fields per line separated by spaces:

| Field | Description |
|-------|-------------|
| 1 | Segment name (filename without the `.wav` or `.png` extension) |
| 2 | Numerical score — the higher the value, the more certain the system is that it is the target person |
| 3 | Hard decision: `1` for the target person, otherwise `0` (assuming a prior probability of 0.5) |

## Required Systems

You are free to choose the programming language or tools used to process the data. Each group must submit at least:

1. **one system based on image data only**,
2. **one system based on audio data only**,
3. **one system combining both modalities** (multimodal fusion).

The chosen method of combining modalities (e.g. at the feature or score level) must be described and justified in the documentation.

## Submission

- You may submit multiple result files, but **at most 6 files will be processed**.
- Result files can be uploaded to IS until **Monday, May 4th, 23:59**.
- The correct answer key will be published on **May 5th**.
- At the final lecture on **May 6th, 2026**, we will analyze your results and solutions.

### Groups and Archive Structure

You will work on this project in groups of **1 to 2 people**, which you can sign up for in IS.

Choose result filenames so that the type of system is recognizable (e.g. `audio_GMM`, `image_linear`).

Each group uploads all result files packed into a single ZIP archive named `login1_login2.zip` or `login1.zip`, depending on group size.

Archive contents:

```
login1_login2.zip
├── *.txt                 (result files)
├── SRC/                  (source code of implemented systems)
└── dokumentace.pdf
```

> **Please do not include the evaluation data in the ZIP archive!**

## Documentation

The `dokumentace.pdf` file, written in Czech, Slovak, or English, will describe your solution. The documentation (**approximately 3 A4 pages**) must contain:

- a description of the designed systems (audio, image, multimodal),
- a description of the validation strategy used,
- the approach to generalization and overfitting prevention.

The documentation must make it possible to understand **why** the system was designed in that particular way, not just **how** it is implemented. Focus on the reasoning leading to the final system design: how you evaluated the systems during development and which techniques or decisions had a positive effect on system performance.

This document will also describe how to reproduce your results using the attached code. Include:

- how to compile the code,
- how to run the systems,
- where to find the output files,
- which external tools need to be installed and used.

## Oral Defense

Part of the evaluation will be a **short oral defense** after project submission, which will take place in the first weeks of the exam period (time slots will be announced). Students should be able to explain the design of their solution and answer questions about the methods used.

## Grading

The full score (**25 points**) will be awarded to a solution that:

- correctly loads data and produces results,
- contains functional systems for image, audio, and their combination,
- uses an appropriate validation strategy,
- addresses the problem of generalization and limited data,
- contains quality documentation enabling reproduction and understanding,
- is successfully presented at the oral defense.

Points will be deducted if some requirements are not met.

