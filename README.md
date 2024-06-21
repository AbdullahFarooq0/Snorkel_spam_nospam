# Snorkel_spam_nospam

This code demonstrates how to use Snorkel, a data-centric AI framework, to develop a machine learning model for classifying YouTube comments as spam or not spam (ham). Basically this technique can be used to check spam/nospam for messages as well.. As they are reviews that are checked for spam and no spam. 

1. **Setup and Data Loading**:
    - Install necessary packages: Snorkel, utility functions, and clone the Snorkel tutorials repository.
    - Load the SPAM dataset using `load_spam_dataset()` from the Snorkel tutorials, which returns training and test dataframes.
    - Extract label vectors (`Y_test`) from the test dataframe.

2. **Defining Constants**:
    - Define constants for labels: `ABSTAIN`, `HAM`, and `SPAM`.

3. **Exploration**:
    - Display and explore sample data from the training dataframe to understand its structure and contents.

4. **Creating Labeling Functions (LFs)**:
    - **Regex-based LF**: Identify spam comments containing the phrase "check out".
    - **TextBlob-based LFs**: Use TextBlob to analyze sentiment and create LFs for polarity and subjectivity.
    - **Keyword-based LFs**: Create several LFs that flag comments based on the presence of certain keywords (e.g., "my", "subscribe", "http", "please", "song").
    - **Short Comment LF**: Classify short comments as ham.
    - **NLP-based LF**: Use spaCy to classify comments mentioning specific people and being short as ham.

5. **Applying LFs**:
    - Apply the LFs to the training and test dataframes using `PandasLFApplier`.
    - Generate label matrices (`L_train` and `L_test`) representing the LF outputs.

6. **Analysis of LFs**:
    - Use `LFAnalysis` to analyze the performance and coverage of the LFs.

7. **Building and Training Label Models**:
    - **Majority Label Voter**: A simple model that predicts the majority vote of the LFs.
    - **Label Model**: A more sophisticated model that learns the accuracy of each LF and combines their outputs.
    - Train the Label Model and generate probabilistic labels for the training data.

8. **Filtering Unlabeled Data**:
    - Filter out unlabeled data points (those with all LFs abstaining) from the training set.

9. **Feature Extraction and Model Training**:
    - Use `CountVectorizer` to convert text data into numerical features (n-grams).
    - Train a logistic regression classifier using the features and filtered probabilistic labels.

10. **Evaluation**:
    - Evaluate the accuracy of both the Majority Label Voter and the Label Model on the test set.
    - Evaluate the logistic regression model on the test set.

11. **Predicting New Data**:
    - Create a new dataset of reviews and transform it using the trained vectorizer.
    - Use the trained logistic regression model to predict whether each new review is spam or ham.
   
# For Email Spam check:
Setup and Data Loading:

Gather a dataset of emails labeled as spam or non-spam (ham). This dataset should be divided into training and test sets.
Defining Constants:

Define constants for labels: ABSTAIN, HAM, and SPAM.
Exploration:

Explore the email dataset to understand its structure and the types of features (e.g., sender, subject, body content) it contains.
Creating Labeling Functions (LFs):

Regex-based LFs: Identify common spam patterns in email content, such as phrases like "free money", "click here", etc.
TextBlob-based LFs: Analyze sentiment if relevant. Spams might often have high polarity (very positive or negative sentiment) and high subjectivity.
Keyword-based LFs: Create LFs that flag emails containing certain keywords often found in spam (e.g., "win", "prize", "urgent").
Other LFs: Consider LFs based on the presence of attachments, links, or unusual email addresses.
Applying LFs:

Apply the LFs to the email training and test datasets to generate label matrices (L_train and L_test).
Analysis of LFs:

Use LFAnalysis to analyze the performance and coverage of the LFs, ensuring they effectively identify spam indicators.
Building and Training Label Models:

Majority Label Voter: Use this model to predict the majority vote of the LFs.
Label Model: Train a label model to learn the accuracy of each LF and combine their outputs.
Filtering Unlabeled Data:

Filter out unlabeled data points (those with all LFs abstaining) from the training set to improve model training.
Feature Extraction and Model Training:

Use CountVectorizer or another suitable method to convert email text into numerical features.
Train a machine learning classifier (e.g., logistic regression, SVM, or neural network) using the features and filtered probabilistic labels.
Evaluation:

Evaluate the accuracy of the Majority Label Voter and the Label Model on the test set.
Evaluate the final classifier (e.g., logistic regression model) on the test set.
Predicting New Emails:

Transform new email texts into features using the trained vectorizer.
Use the trained classifier to predict whether each new email is spam or ham.

**Summary**:
The code demonstrates a complete workflow for using Snorkel to label and train a spam detection model for YouTube comments. It includes defining labeling functions, training a label model to combine LF outputs, filtering data, extracting features, and training a final logistic regression classifier. The model is then evaluated for accuracy and used to predict new comments.
