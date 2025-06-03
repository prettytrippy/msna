# Code for "A Robust Deep Learning Framework for Detecting Bursts in Muscle Sympathetic Nerve Activity"


## Abstract

Muscle sympathetic nerve activity (MSNA) is a key physiological signal that provides insights into the functioning of the sympathetic nervous system. Characterized by bursts of neural activity, MSNA signals play a crucial role in understanding both normal and pathological states. Accurately detecting these bursts is essential for quantitative analysis and deeper exploration of sympathetic nerve dynamics. However, the tedious task of detecting bursts is currently performed by trained experts, leading to potential burnout and increased risk of error. In this study, we present a novel machine learning-based burst detection method that combines integrated MSNA activity and electrocardiography activity in a convolutional neural network to robustly identify burst peaks. Our approach achieves an average F1 score of 0.87$\pm$0.03 in detecting expert-annotated bursts in a dataset including resting autonomic nervous system recordings of 41 healthy female participants when evaluated under a five-fold cross-validation. Our approach outperformed several alternative methods including some previously published automated burst detection approaches.

## Setup

`git clone https://github.com/prettytrippy/msna`
`cd msna`
`pip install -e .`

## Application

`cd msna/app`
`python3 detect_bursts.py`

This application allows a user to select a LabChart text file, and add automated burst annotations to it. The file is saved to a selected directory, and outcome measure information is printed to the terminal.