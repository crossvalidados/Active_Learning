# Active Learning 

In this python notebook, different Active Learning strategies combined with diversity criteria are compared in order to find the best set of labeled observations to train a SVM classifier.

## Problem description

The [Semeion Handwritten Digit Data Set](https://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit) is a database of handwritten digits. Each record represents a handwritten digit, orginally scanned with a resolution of 256 grays scale (28). The goal is to obtain the best possible result by minimizing the number of labeled observations used to train a classification model by selecting samples that provide the maximum information possible.

## Active Learning strategies

- MS (margin sampling)
- MCLU (multi-class label uncertainty)
- SSC (significance space construction)
- nEQB (normalized entropy query bagging)

## Diversity criteria 
- MAO (most ambiguous and orthogonal)
- MAO lambda
- diversity by clustering.

## Results

![Results](Results.png)



