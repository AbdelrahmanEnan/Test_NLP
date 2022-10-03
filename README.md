# Clustering_Test_NLP
Spooky Author Identification; Testing k-means algorithms against other ML classification algorithms.


# Introduction
The aim of this paper is to build models that can identify the author of small unknown text (sentence) after converting it to a set of variables (ex. number of words, number of letters, etc…). Some classical algorithms are used like K-NN, Naïve Bayes, and K-means, and their results are compared to results obtained using more advanced methods (machine learning models) like Random Forest, and Support Vector Machine.
> **Note**
> 
> All methods except K-means are in [NLP.R](https://github.com/AbdelrahmanEnan/Clustering_Test_NLP/blob/main/NLP.R) file.
> 
> K-means algorithm is built from scratch for educational purposes in [clustering.py](https://github.com/AbdelrahmanEnan/Clustering_Test_NLP/blob/main/clustering.py)

The goal was accomplished by data pre-processing and generating new features, followed by training models on dataset, then testing models performance, and finally model improving using different techniques.

# Dataset
## Data Acquisition
The dataset [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification) is a public dataset, that was uploaded on Kaggle as a competition to predict the author of excerpts from horror stories by Edgar Allan Poe, Mary Shelley, and Howard Phillips Lovecraft.
## Initial Dataset
Initially, dataset was consisting of 19579 observations with 3 attributes which are “id”, “text”, and “author” attributes. One output attribute (‘author’), one unimportant attribute (‘id’) which will be droped later, and one input (‘text') which will be processed to obtain more attributes.
## Data Pre-processing
### Features Generation and Feature Removal
As stated before, (‘id’) attribute is removed in this stage; as it is just unimportant data for model training.

Input attribute was processed to extract more meaningful attributes for computer to learn from. The generated attributes include:
1. number of word in the text.
2. number of letters in the text.
3. average number of letters text’s word.
4. Part of Speech Tagging (17 attributes, each holds count of number of words for the aimed tag).
5. Occurrence of most frequent words in the text (1100 attributes, each holds count of number of the specific word in the text).
> **Note**
> 
> The most frequent 1100 words are taken (arbitrary) to be attributes for each observation, this number can be change or even to take all words. 
> 
> If we take all words to be attributes, the dataset attributes will be in form of the well-known ("words unigram") methods for Natural Language Processing NLP.
### Data Normalization
When dealing with distance-based methods (ex. K-NN and SVM), or when applying Regulaization it is necessary to normalize data in order to avoid data redundancy or imposing fake weight due to different ranges of data’s variables.

The “Max-min” normalization technique was the chosen method over z-score method in this document.
### Data split
Dataset was split so that 80-20% “Train set”-“Test set”. Split was performed randomly, to ensure that the subsets have the same representation of output classes as the original dataset.
## Data Description (after pre-processing)
• No nulls.
• Weak correlation among features and output.
• Numeric input features.
• Factor class output.
• 1138 input attributes.
• Training set of 15663 observations.
• Testing set of 3916 observations.
# Algorithms
1. K-NN
2. Naive Bayes
3. Random Forest
4. Support Vector Machine
5. Neural Networks
6. K-means

# Results
|     Model                     |     Parameter(s)      |     Value(s)     |     Accuracy    |     Kappa    |
|-------------------------------|-----------------------|------------------|-----------------|--------------|
|     K-Means                   |     k                 |     3            |     41.0        |      -       |
|     K-NN                      |     k                 |     5            |     51.3        |     0.24     |
|     K-NN                      |     k                 |     25           |     52.4        |     0.24     |
|     K-NN                      |     k                 |     39           |     50.1        |     0.19     |
|     Naïve Bayes               |     -                 |     -            |     63          |     0.4      |
|     Random Forest             |     nTrees,   mtry    |     500,   33    |     68          |     0.5      |
|     Support Vector Machine    |     C, sigma          |     1, 0         |     76          |     0.63     |

As shown above, Support Vector Machine with the default parameters has the best accuracy over all other algorithms 
# Model Improving (Tuning)
After getting the results of all models, Support Vector Machine with the default parameters showed a high preformance, and now it is the time to improve the result by tuning the parameters, anfd introducing Cross-Validation to avoid overfitting.

The model was tuned by trying different valyes for the following parameters:
1. Different kernel: (rbfdot, polydot, laplacedot, etc.)
2. Different types: (C-svc, spoc-svc, and kbb-svc)
3. Parameter tuning: (C and sigma Parameters)
4. Introducing random k Cross-Validation (5 folds), to validate the results. An accuracy of 77.9% were obtained and kappa = 0.65

# Conclusion 
The best model was Support Vector Machine of type “spoc-svc”, kernel “rbfdot”, C-parameter = 5, and sigma-parameter = 0, using 5-folds Cross-validation. And the results were accuracy of 78%.

# Furthr Work
Selecting different set of features (word bi-gram for example) need to be investigated as it might result a more accurate models. 

Further tuning is needed for Random Forest and Neural Networks models, since tuning its two parameters (ntree and mtry) may result a better accuracy. 

Trying different model that is not used in this paper is a topic for further work as well.
