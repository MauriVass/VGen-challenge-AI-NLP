#!/usr/bin/env python
# coding: utf-8

# ## Libs

import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords as sw
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import pickle


# ## IMPORT DATA

print('---IMPORT DATA---')
train_ds = pd.read_csv('train_set.csv')
# print(f"Length train set: {len(train_ds)}")
test_ds = pd.read_csv('test_set.csv')
# print(f"Length test set: {len(test_ds)}")


# ## PREPROCESS

print('---PREPROCESS---')
#Found some Deutsch job offers. Removed since they are outliers
#Hard removing them
remove_jobs = [12, 32, 569, 834, 893, 1256, 1261]
train_ds = train_ds.drop(remove_jobs)


#Download if needed
#nltk.download('stopwords')

stop_words = sw.words('italian')
stop_words_eng = sw.words('english')

#Common verbs according to Google
commonverbs = ['abbandonare','abitare','accadere','accendere','accettare','accogliere','accompagnare','accorger','addire','affascinare','affermare','affrontare','aggiungere','aiutare','allontanare','alzare','amare','ammazzare','ammettere','andare','annunciare','apparire','appartenere','appoggiare','aprire','armare','arrestare','arrivare','ascoltare','aspettare','assicurare','assistere','assumere','attaccare','attendere','attraversare','aumentare','avanzare','avere','avvenire','avvertire','avvicinare','baciare','badare','bastare','battere','bere','bisognare','bruciare','buttare','cadere','cambiare','candidare','camminare','cantare','capire','capitare','celebrare','cercare','chiamare','chiedere','chiudere','colpire','cominciare','compiere','comporre','comprendere','comprare','concedere','concludere','condurre','confessare','conoscere','consentire','conservare','considerare','contare','contenere','continuare','convincere','coprire','correre','costituire','costringere','costruire','creare','credere','crescere','dare','decidere','dedicare','descrivere','desiderare','determinare','dichiarare','difendere','diffondere','dimenticare','dimostrare','dipendere','dire','dirigere','discutere','disporre','distinguere','distruggere','diventare','divenire','divertire','dividere','domandare','dormire','dovere','durare','elevare','entrare','escludere','esistere','esporre','esprimere','essere','estendere','evitare','ferire','fermare','figurare','finire','fissare','fondare','formare','fornire','fuggire','fumare','gettare','giocare','girare','giudicare','giungere','godere','gridare','guardare','guidare','immaginare','imparare','impedire','importare','imporre','incontrare','indicare','iniziare','innamorare','insegnare','insistere','intendere','interessare','invitare','lanciare','lasciare','lavorare','legare','leggere','levare','liberare','limitare','mancare','mandare','mangiare','mantenere','mondare','meritare','mettere','morire','mostrare','muovere','nascere','nascondere','notare','occorrere','occupare','offendere','offrire','opporre','ordinare','organizzare','osservare','ottenere','pagare','parere','parlare','partecipare','partire','passare','pensare','perdere','permettere','pesare','piacere','piangere','piantare','porre','portare','posare','possedere','potere','preferire','pregare','prendere','preoccupare','preparare','presentare','prevedere','procedere','produrre','promettere','proporre','provare','provocare','provvedere','pubblicare','raccogliere','raccontare','raggiungere','rappresentare','recare','rendere','resistere','restare','ricevere','richiedere','riconoscere','ricordare','ridere','ridurre','riempire','rientrare','riferire','rifiutare','riguardare','rimanere','rimettere','ringraziare','ripetere','riportare','riprendere','risolvere','ricercare','rispondere','risultare','ritenere','ritornare','ritrovare','riunire','riuscire','rivedere','rivelare','rivolgere','rompere','salire','saltare','salutare','salvare','sapere','sbagliare','scappare','scegliere','scendere','scherzare','scomparire','scoppiare','scoprire','scorrere','scrivere','scusare','sedere','segnare','seguire','selezionare','sembrare','sentire','servire','significare','smettere','soffrire','sognare','sorgere','sorprendere','sorridere','sostenere','spegnere','sperare','spiegare','spingere','sposare','stabilire','staccare','stare','stringere','studiare','succedere','suonare','superare','svegliare','svolgere','tacere','tagliare','temere','tendere','tenere','tentare','tirare','toccare','togliere','tornare','trarre','trascinare','trasformare','trattare','trovare','uccidere','udire','unire','utilizzare','usare','uscire','valere','vedere','vendere','venire','vestire','viaggiare','vincere','vivere','volare','volere','volgere','voltare','livellare','affiancare','fare','macchinare','abbattere','agevolare','sfidare','accreditare','pressare','ambire','respirare']
#Words that can be not too useful (either too common or less common) found in the training set
not_useful_words = ['italia','dio','etc','eeo','iva','poi','dopo','ora','acqua','abb','teamfähig','abteilung','dafür','age','religion','sex','allâ','milano','torino','anno','già','nonché','asap','sede','autobus','altro','able','fonte','add','piano','africo','alcune','almeno','cosa','strumento','comau','percorsostruttura','accordare','ministero','hejsberg','obbligo','konzepten','affinchè','essi','affine','nazionale','tesys','agevolazione','tendenza','konzepten','alare','migliaio','alcuno','caso','allinterno','dellazienda','realtà','altresì','mago','ambizione','america','asia','fino','accreditato','contesto','strutturato','neben','der','erneuerbarer','altamente','ambo','sesso','amichevole','specifico','suchen','verstärkung','sud','suchen','buonopatenti','trasferta','estero','settimanale','occupera','messico','volto','www','wort','schrift','zuschuss','œautorizzo','zuora','varo','santo','zona','com','well','weiteren','wec','ausbau','erneuerbaren','welcoming','weiteren','ospedaliero','œregolamento','qui','vita','vitae','dellâ','kenntnisse','modena','umano','reggio','ove','kafka','agosto','udine','personaliâ','diversamente','verra','chieti','sitowww','kformazione','bandr','dijunior','voghera','grottte','presto','lavoratrice','catia','gamba','giorco','tranne','libro','biblioteca','palermo','teamoffriamo','tredicesimo','emc²','esposto','saperejava','confà','ops','descriptiono','liceità','ainformatica','sottolineare','englisch','laddove','every','polonia','precison','minuto','singolo','unâ','deutsch','besitzen','sicher','bereich','città','inviarci','lecce','kubernetes','reggio','emilia','modena','bologna','napoli','mese','giorno','ultimo','ciclo','divertirti','roma','team','convolte','collaborarai','timing','budapest','sansepolcro','valtiberina','mondosi','rivolgendosi','planet','abbastanza','abilità']

#Words that can be useful
include_words = ['bene','bel','buon','r&d','ar','vr']
for i in include_words:
    if i in stop_words:
        stop_words.remove(i)


stop_words= stop_words+stop_words_eng+not_useful_words+commonverbs
stop_words = set(stop_words)
include_words = set(include_words)

#May need to download this file (it is 571 MB):
#python -m spacy download it_core_news_lg
lemmatizer = spacy.load("it_core_news_lg", disable=["tagger", "parser", "ner"])
def CleanSentence(sentence,sw=stop_words):
    sentence = sentence.lower()
    #Remove all non-alphanumeric character, excluding '&' (like: R&D)
    sentence = re.sub("[^\w&-]+|_|\d+", " ", sentence)
    sentence = re.sub("-", "", sentence)
    
    lemmas = lemmatizer(sentence)
    newSentence = ""
    min = 3
    max = 16
    for lemma in lemmas:
        word = lemma.lemma_
        if word not in stop_words:
            if  (min <= len(word) <= max or word in include_words):
                newSentence = newSentence + word + " "
    return newSentence

def CleanText(text):
    sentences = []
    for row in text:
        sentences.append(CleanSentence(row))
    
    return sentences


train_ds['clean'] = CleanText(train_ds['Job_offer'])
test_ds['clean'] = CleanText(test_ds['Job_offer'])

#Test preprocessing step
# ind = 123
# print(train_ds['Job_offer'][ind])
# print(train_ds['clean'][ind])


#Calculate TF-IDF

column = 'clean'
def X_tfidf(sentences):  
    tfidf = TfidfVectorizer(min_df=2, max_df=0.2, ngram_range=(1,3),lowercase=True)#,strip_accents='ascii'
    X = tfidf.fit_transform(sentences)
    return X, tfidf

train_vec, vectorizer = X_tfidf(train_ds[column])
test_vec = vectorizer.transform(test_ds[column])

X_train, y_train, X_test, y_test = train_vec, train_ds['Label'], test_vec, test_ds['Label']

# ## MODEL

def get_score(classifier,X_test,y_test):
    y_pred = classifier.predict(X_test)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1_score_ = f1_score(y_test, y_pred, average='macro')
    print(f"#Precision: {precision:.5f}")
    print(f"#Recall: {recall:.5f}")
    print(f"#f1 Score: {f1_score_:.5f}")
    return precision,recall,f1_score_,y_pred

def train_model(classifier, X_train, X_test, y_train, y_test, printAll=False):
    
    classifier.fit(X_train, y_train)
    
    precision,recall,f1_score_,y_pred = get_score(classifier,X_test,y_test)
        
    if(printAll):
        print(confusion_matrix(y_test,y_pred))      
        print(classification_report(y_test,y_pred))
    return precision,recall,f1_score_,y_pred

print('---TRAINING---')
lr = LogisticRegression(penalty='l2',max_iter=10**9,C=4.1,random_state=10, solver='liblinear')
_,_,_,y_pred = train_model(lr, X_train, X_test, y_train, y_test)


# ## SAVE PREDICTIONS

print('---SAVE PREDICTIONS---')
columns = ['Job_description', 'Label_true', 'Label_pred']
df = pd.DataFrame(list(zip(test_ds['Job_offer'],y_test,y_pred)),columns=columns)
df.to_csv('prediction.csv',sep=';',index=False)


# ## READ PREDICTIONS

print('---READ PREDICTIONS---')
predictions_ds = pd.read_csv('prediction.csv', sep=';', header=0)


# ## SAVE MODEL

print('---SAVE MODEL---')
filename = 'model.sav'
pickle.dump(lr, open(filename, 'wb'))


# ## LOAD MODEL

print('---LOAD MODEL---')
filename = 'model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# ## TEST LOADED MODEL

print('---TEST LOADED MODEL---')
_ = get_score(loaded_model,X_test,y_test)

