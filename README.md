# VGen-challenge-AI-NLP

Artificial Intelligence challenge on Natural Language Processing organized by VGen and Randstad Technology. [Link Challenge](https://www.vgen.it/hackathon/randstad-artificial-intelligence-challenge/) \
The main goal was, given a training set, to predict the label of some job offers given their job description.
The labels are 5 and the are:
- Java Developer,
- Web Developer,
- Programmer,
- System Analyst,
- Software Engineer.

Example: \
**Job description**: *"Siamo alla ricerca di uno SVILUPPATORE BACKEND JAVA, da inserire all'interno di una società che punta all'innovazione per i propri clienti e partner., * Sviluppo di soluzioni e applicazioni in un contesto Enterprise"*; \
**Label**: *Java Developer*

## PREPROCESS
Steps used for the preprocessing part:
- Removed not italian and english job descriptions;
- Removed most common italian and english words;
- Removed numbers and non-alphanumeric characters;
- Removed too short or too long words;
- Calculate TF-IDF of the words.

## TRAINING
Many models were tested but the most performing one was the **Logistic Regression** classifier. \
The metrics to be considered were:
- Precision,
- Recall,
- F1-score.

## OTHER PARTS
To be complaint with the challenge's requiremnts some more fuctions were added:
- SAVE PREDICTIONS: save the classifier's predictions of the test set in a .csv file.
- READ PREDICTIONS: read the predictions from the file saved with the previous fuction.
- SAVE, LOAD and TEST MODEL: used to save, load and test the classifier.
