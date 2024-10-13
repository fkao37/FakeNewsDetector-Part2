# FakeNewsDetector-Part2
This project is Part 2 of the "Fake News Detector," focusing on comparing the performance of various machine learning models—Logistic Regression, SVM, Random Forest, XGBoost, and GradientBoost—with "Deep Learning Neural Network Models" such as RNN/LSTM/GRU and RNN/BiDirectional LSTM/GRU layers. CUDA GPU acceleration is utilized to enhance model performance where applicable.

Additionally, a BERT Transformer (not covered in detail) is employed to summarize each article, creating a standardized writing style across the dataset. A 'mix-and-match' strategy is then applied to evaluate the models based on the writing styles used during training and testing, providing insights into their effectiveness under different conditions.
# Repository Structure
	./data_fakenews/
        True.csv                            ==> curated true news articles
        Fake.csv                            ==> curated fake news articles 

    ./model_results/
        model_performance_sum_sum.xlsx      ==> results of training with summary, testing with summary
        model_performance_sum_txt.xlsx      ==> results of training with summary, testing with text
        model_performance_txt_sum.xlsx      ==> results of training with text, testing with summary
        model_performance_txt_txt.xlsx      ==> results of training with text, testing with text

    configuration file
        FakeNewsModel_txt_txt.ini           ==> configuration for training with text, testing with text
        FakeNewsModel_txt_sum.ini           ==> configuration for training with text, testing with summary
        FakeNewsModel_sum_txt.ini           ==> configuration for training with summary, testing with test
        FakeNewsModel_sum_sum.ini           ==> configuration for training with summary, testing with summary

    Jupyter Notebook Model Files
        fn_detector_cutpub_txt_txt.ipynb    ==> jupyter notebook model file for training with text, testing with text
        fn_detector_cutpub_txt_sum.ipynb    ==> jupyter notebook model file for training with text, testing with summary
        fn_detector_cutpub_sum_txt.ipynb    ==> jupyter notebook model file for training with summary, testing with text
        fn_detector_curpub_sum_sum.ipynb    ==> jupyter notebook model file for training with summary, testing with summary
# Introduction
Fake news, also known as misinformation or disinformation, refers to the deliberate spread of false or misleading information designed to deceive with the intent to manipulate for financial gains and to distort public perception. Amplified by today’s speed and reach of social media platforms, blogs, and even mainstream media, fake news infiltrates every corner of society, influencing public opinion, political outcomes, and even personal decisions. Its ease of access and ability to spread rapidly make it a significant challenge for governments, businesses, and individuals alike.

The authors behind fake news often have nefarious intentions, ranging from political manipulation, financial gain, to simply creating chaos. These actors could be malicious individuals, groups, or even state-sponsored entities that use fake news to sway elections, manipulate stock markets, incite violence, or damage the reputations of people or organizations. They prey on people's emotions, targeting topics that spark outrage, fear, or strong opinions. Unfortunately, fake news can spread much faster than factual corrections, leading to widespread damage before the truth is revealed.

Fake news can cause severe harm to businesses. For example, false information about a company’s financial health, leadership, or products can lead to a loss of consumer trust, a drop in stock prices, or even legal complications. Businesses rely on accurate information to make strategic decisions, but disinformation can distort market trends or consumer behavior. Furthermore, a company’s reputation can be tarnished, leading to customer attrition, partnerships falling apart, and a tarnished brand image. In extreme cases, fake news can even lead to boycotts or protests against the business, causing long-term financial and operational challenges.

The broader world also faces significant threats from fake news. False information can lead to public unrest, political instability, and polarization of societies. It can be weaponized to destabilize governments, spark international conflicts, and fuel extremism. Health misinformation, such as false claims about vaccines or treatments, can result in widespread public health crises, as seen during the COVID-19 pandemic. The spread of conspiracy theories and unfounded claims undermines public trust in institutions, experts, and scientific facts, eroding the foundation of truth upon which society depends.

Implementing robust fake news detection systems can play a crucial role in combating this growing menace. In the business context, such detectors can protect brand reputation, monitor the spread of false information, and quickly counteract harmful narratives with facts. This could lead to improved customer loyalty, business continuity, and protection of market positions. Detecting and addressing fake news early can help prevent costly crises and enable businesses to act swiftly before significant damage is done.

For humanity, fake news detection can safeguard democratic processes, enhance social cohesion, and promote an informed society. By identifying false information before it gains traction, these systems can reduce the spread of harmful ideologies, improve public trust, and ensure that critical societal decisions are made based on facts rather than manipulation. The ability to filter out disinformation helps create a more transparent, educated world where people and businesses alike can thrive on the foundation of truth and reliability.

## Challenges
Building a robust and highly resilent fake new detector faces serious challenges in:
#### High-Dimensional Data
Text data, especially after vectorization with methods like TF-IDF, can result in very high-dimensional feature spaces, significantly increasing computation time, memory usage, and makes model training process impossible slow.
#### Overfitting Due to Writing Style Differences
Overfitting occurs when a machine learning model performs exceptionally well on the training data but struggles to generalize to unseen or test data. In the context of fake news detection, overfitting can lead to the model memorizing specific patterns or phrases from the training set instead of learning generalized language patterns for detecting misinformation.
#### Dynamic and Evolving Language
New words, memes, and trends appear regularly, making it hard for static models to remain effective.
#### Imbalance in the Dataset
Imbalance number of training sample for true and fake introduce hidden bias causing classification error
## Machine Learning and Neural Networks
Machine learning and neural networks are powerful tools for combating fake news, thanks to their ability to process large datasets, detect complex patterns, and adapt to evolving misinformation. Machine learning models identify subtle linguistic cues that distinguish real from fake news, while neural networks, especially deep learning and transformers, excel at understanding the structure and context of language. Their scalability enables real-time detection, protecting businesses from reputational damage and ensuring society receives reliable information.

# Fake News Detector
This project builds on the data exploration from [Part 1](https://github.com/fkao37/FakeNewsDetector) by developing and evaluating various machine learning and deep learning models on the curated dataset. TF-IDF (Term Frequency-Inverse Document Frequency) vectorization serves as the primary word tokenization technique throughout the pipeline. A key focus is comparing the models' robustness and resilience against diverse writing styles while capturing nuances across different forms of expression. This includes developing a BERT transformer (not covered in this project) to pre-process each article, producing condensed summaries that can be used for training and testing. Mix-matching each article’s full text with its condensed version tests the model’s robustness. The models' classification accuracy is assessed to detect signs of overfitting, ensuring a balance between performance and generalization. Where applicable, CUDA/GPU is enabled to accelerate models that support it.

This project have the following objectives:
* Develop a model that is robust enough to classify between different article writing styles, and language usages
* Efficient model performance, leveraging GPU hardware acceleration where applicable
* Model improvement with incremental updates with new training data 
* Compare and evaluate various Machine Learning and Neural Network Models

## DataSet
A curated [SimpliLearn Fake News Dataset](https://www.simplilearn.com/tutorials/machine-learning-tutorial/how-to-create-a-fake-news-detection-system) consists of a set of 2 data files: 
* True.csv  ==> true news, 21,417 articles 
* Fake.csv  ==> fake news, 23,481 articles 
### Pre-processing
Previous [EDA analysis](https://github.com/fkao37/FakeNewsDetector) of the dataset revealed that the news source or publisher is uniquely embedded at the start of each true news article, separated by a '-' character before the start of the article text. This 'publisher' embedding is not present in the fake news data set. Therefore,  to ensure the model comparison only each article's text patterns:
* The publisher reference are removed from each data article
* Unwanted escape sequences from urls are removed
* All articles are normalized and converted to lower case
### BERT Features
Bert transformer is used to create additional features derived from each article. Asides from using a different writing style to create the article 'summary', the others feature each represents a different aspect of each of the news article.
### Data Balance
An equal number of randomly selected records is selected from the larger dataset (fake news) to create a balanced ratio of true and fake news dataset. A total of 21,417-records is randomly selected from the Fake.csv.
### Model Dataset
The new combined dataset is used as the bases for all the subsequent training and testing data creation.<br>

Features (original):<br>
* title         ==> text string, headline of news article<br>
* subject       ==> text string, type of news: news, politics, goveronment news, ...<br>
* text          ==> text string, news article<br>
* date          ==> text string date format of the date referenced by the news articles<br>
<br>
Added Features (bert):<br>
* summary       ==> text string, condensed summary of news article<br>
* sentiment     ==> text string, negative, positive<br>
* emotion       ==> text string, anger, fear, joy, ...<br>
* intent        ==> neutral, entertainment, ...<br>
* assertions    ==> no_claim, claim_detected,...<br>
<br>
Added Features (model):<br>
* class         ==> true / fake classification<br>

A new pandas data frame with only feature columns: text, summary, and class is created and used as the base for all subsequent datasets for model creation.

## GPU Hardware Acceleration
The current platform supports a Nvidia RTX4060 GPU.  Proper hardware support drivers and appropriate tensorflow library installed to enable GPU support.  CuML support is not considered and out of scope for this project.  As a result, any machine learning and neural network models that uses the tensorflow library can take advantage of the GPU acceleration.

## Training / Testing Data Split
A 65% Training data, 35% Testing data split is used to divide the combined dataset of 2x21,417 records for model training and testing.

## TF-IDF Vectorization, Word Embedding
TF-IDF initialized with max_features=5000 encodes number of occurrence and appearance frequency in each of the articles 'text' and 'feature' columns.  The fit and transformed TF-IDF object is used as the source for all the datasets used later in the pipeline.  A mix-match of training and testing datasets are used to explore the different model performances:
* Training: text, Testing: text     ==> Same writing style
* Training: text, Testing: summary  ==> Different writing style
* Training: summary, Testing: text  ==> Different writing style
* Training: summary, Testing: summary ==> Same writing style

## Model Dataset
In addition the normal training and testing datasets, additional dense matrix based, and shaped matrix based datasets are created for specific model use where applicable.

## Model Creation / Testing
Each classification model is then created, trained and tested and accuracy performance captured.

# Results
### Machine Learning Models CPU Only (n_estimators=10)
* Random Forest
* AdaBoost
* Gradient Boost
* Logistic Regression (GridSearch)
* Support Vector Machines (GridSearch)
### Machine Learning Models GPU Accelerated
* XGBoost (max_depth=4)
### Neural Network Models (epochs=20, batch_size=128)
* RNN + LSTM (64-neurons) + GRU(32-neurons)
* RNN + BiDirectional LSTM(64-neurons) + BiDirectional GRU(32-neurons)

####  Training: Text - Testing: Text  ####
| Model                          | Training Time   | Testing Time   |   Accuracy |   Precision(weighted) |   Recall(weighted) |   f1-score(weighted) |
|:-------------------------------|:----------------|:---------------|-----------:|----------------------:|-------------------:|---------------------:|
| rnn_bidirectional_detector.h5  | 43.38 sec       | 2.29 sec       |     0.9844 |                0.9844 |             0.9844 |               0.9844 |
| rnn_unidirectional_detector.h5 | 26.74 sec       | 1.61 sec       |     0.9834 |                0.9834 |             0.9834 |               0.9834 |
| randforest_detector.h5         | 5.11 sec        | 0.20 sec       |     0.9582 |                0.9587 |             0.9582 |               0.9582 |
| adaboost_detector.h5           | 26.79 sec       | 1.28 sec       |     0.9252 |                0.9253 |             0.9252 |               0.9251 |
| gradboost_detector.h5          | 65.83 sec       | 0.18 sec       |     0.8943 |                0.8944 |             0.8943 |               0.8943 |
| xgboost_detector.h5            | 2.46 sec        | 0.10 sec       |     0.9817 |                0.9817 |             0.9817 |               0.9817 |
| SVMGridSearch                  | 5001.85 sec     | 109.88 sec     |     0.9817 |                0.9884 |             0.9884 |               0.9884 |
| Logistic Regression GridSearch | 13.67 sec       | 0.01 sec       |     0.9817 |                0.9857 |             0.9857 |               0.9857 |

####  Training: Summary - Testing: Text  ####
| Model                          | Training Time   | Testing Time   |   Accuracy |   Precision(weighted) |   Recall(weighted) |   f1-score(weighted) |
|:-------------------------------|:----------------|:---------------|-----------:|----------------------:|-------------------:|---------------------:|
| rnn_bidirectional_sum_txt.h5   | 45.44 sec       | 2.34 sec       |     0.8985 |                0.9051 |             0.8985 |               0.898  |
| rnn_unidirectional_sum_txt.h5  | 31.43 sec       | 1.68 sec       |     0.9037 |                0.9099 |             0.9037 |               0.9032 |
| randforest_sum_txt.h5          | 5.13 sec        | 0.22 sec       |     0.6175 |                0.7749 |             0.6175 |               0.5507 |
| adaboost_sum_txt.h5            | 25.39 sec       | 1.39 sec       |     0.6776 |                0.7326 |             0.6776 |               0.6558 |
| gradboost_sum_txt.h5           | 59.41 sec       | 0.18 sec       |     0.5669 |                0.7063 |             0.5669 |               0.474  |
| xgboost_sum_txt.h5             | 2.17 sec        | 0.10 sec       |     0.6439 |                0.7823 |             0.6439 |               0.5917 |
| SVMGridSearch                  | 2925.44 sec     | 75.27 sec      |     0.6439 |                0.9021 |             0.8821 |               0.8804 |
| Logistic Regression GridSearch | 11.56 sec       | 0.01 sec       |     0.6439 |                0.9588 |             0.9567 |               0.9566 |

####  Training: Text - Testing: Summary  ####
| Model                          | Training Time   | Testing Time   |   Accuracy |   Precision(weighted) |   Recall(weighted) |   f1-score(weighted) |
|:-------------------------------|:----------------|:---------------|-----------:|----------------------:|-------------------:|---------------------:|
| rnn_bidirectional_txt_sum.h5   | 42.01 sec       | 2.17 sec       |     0.9207 |                0.9207 |             0.9207 |               0.9207 |
| rnn_unidirectional_txt_sum.h5  | 27.74 sec       | 1.50 sec       |     0.9173 |                0.9174 |             0.9173 |               0.9173 |
| randforest_txt_sum.h5          | 5.80 sec        | 0.21 sec       |     0.8559 |                0.8569 |             0.8559 |               0.8558 |
| adaboost_txt_sum.h5            | 28.17 sec       | 1.33 sec       |     0.7961 |                0.7986 |             0.7961 |               0.7956 |
| gradboost_txt_sum.h5           | 69.92 sec       | 0.19 sec       |     0.7554 |                0.7605 |             0.7554 |               0.7542 |
| xgboost_txt_sum.h5             | 23.29 sec       | 0.10 sec       |     0.8465 |                0.8478 |             0.8465 |               0.8463 |
| SVMGridSearch                  | 4882.10 sec     | 64.38 sec      |     0.8465 |                0.9354 |             0.935  |               0.9349 |
| Logistic Regression GridSearch | 12.32 sec       | 0.01 sec       |     0.8465 |                0.907  |             0.9006 |               0.9002 |

####  Training: Summary - Testing: Summary  ####
| Model                          | Training Time   | Testing Time   |   Accuracy |   Precision(weighted) |   Recall(weighted) |   f1-score(weighted) |
|:-------------------------------|:----------------|:---------------|-----------:|----------------------:|-------------------:|---------------------:|
| rnn_bidirectional_sum_sum.h5   | 42.21 sec       | 2.24 sec       |     0.9536 |                0.9538 |             0.9536 |               0.9536 |
| rnn_unidirectional_sum_sum.h5  | 30.57 sec       | 1.65 sec       |     0.9549 |                0.955  |             0.9549 |               0.9549 |
| randforest_sum_sum.h5          | 4.96 sec        | 0.21 sec       |     0.9071 |                0.9071 |             0.9071 |               0.9071 |
| adaboost_sum_sum.h5            | 25.54 sec       | 1.35 sec       |     0.8316 |                0.8337 |             0.8316 |               0.8314 |
| gradboost_sum_sum.h5           | 62.37 sec       | 0.21 sec       |     0.8332 |                0.8343 |             0.8332 |               0.8331 |
| xgboost_sum_sum.h5             | 23.29 sec       | 0.11 sec       |     0.9348 |                0.9348 |             0.9348 |               0.9348 |
| SVMGridSearch                  | 2739.28 sec     | 54.99 sec      |     0.9348 |                0.9647 |             0.9646 |               0.9646 |
| Logistic Regression GridSearch | 8.08 sec        | 0.00 sec       |     0.9348 |                0.957  |             0.9568 |               0.9568 |

# Conclusions
* Partial or incremental model training is only available for the newer neural network models, and not consistently available for all of the machine learning models.  Due to this, one cannot reliably compare model performance between each incremental updates, and the results are omitted in the discussion.  However, neural network models' weights can be incrementally updates to increase samples trained, and very stuitable for fake news or fact checker systems going forward.<br>
<br>
* All models performed very well when detecting word patterns with the same writing style due to potential overfitting with the dataset from an single Reuters data stream.  This project used a "Mix-match" training / testing strategy using Bert to generated a condensed article summary with a different writing style.  All models except the neural network based RNN models' classification encountered a significant drop in model classification performance except for the neural network based RNN models.<br>
<br>
* Model with tensorflow libraries enjoyed a significant performance boost in terms of training time requirement when receiving hardware GPU acceleration, otherwise an unbearable long processing time is needed.  Fake news datasets often contain thousands of articles with complex text, requiring significant computation to process. GPU-accelerated parallel processing speeds up matrix operations, reducing the time needed for training from hours or days to minutes or hours.  Using GPUs allows developers to build more accurate and complex models faster, improving both training time and inference speed. As a result, a fake news detector becomes scalable and responsive, which is crucial for real-time applications where quick identification of misinformation is essential.<br>
<br>
* Neural Network based RNN models showed the best consistency across all test cases, even in cases exposing the over fitting of all the other machine learning models. Neural network models are well-suited for fake news detection as they effectively capture the non-linear nature of language in text articles. RNN-based models with LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Units) are well-suited for fake news detection because they excel at understanding sequential dependencies in text, which is crucial for analyzing article structure, context, and meaning. <br>
<br>
* Bidirectional LSTM and GRU models outperform their unidirectional counterparts by leveraging both past and future context, enhancing the ability to process long texts, complex sentence structures, and nuanced language. This holistic approach results in more accurate predictions, making the model more robust in distinguishing between legitimate and fake news.
* This project addressed challenges raised by existing systems that favored XGBoost, Random Forest, and SVM implementations:  
a) significant computational resources demand and 
b) RNN + LSTM models are prone to overfitting. <br>
To overcome these issues, this project employed a BERT transformation of article data, converting it into a standardized "uniform" writing style that effectively normalized variations in writing across different sources and limited combination of potential parameters by a single writing source.<br>
<br>
The BERT transformation process introduced a 1-3 second processing overhead on the current system, which is expected to decrease with larger and more advanced hardware. <br>
<br>
* For systems deployed with Random Forest, XGBoost, SVM, the project showed that best performance when using the full article text as the training source, and a condensed summary as the testing source, while experiencing some classification performance degradation.
