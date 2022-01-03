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
Steps used for the pre-processing part:
- Removed not Italian and English job descriptions;
- Removed most common Italian and English words;
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
To be compliant with the challenge's requirements some more functions were added:
- SAVE PREDICTIONS: save the classifier's predictions of the test set in a .csv file.
- READ PREDICTIONS: read the predictions from the file saved with the previous function.
- SAVE, LOAD and TEST MODEL: used to save, load and test the classifier.

