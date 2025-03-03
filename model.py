import pandas as pd
import spacy
import time
import math
import re

class Sentiment_Analysis():
    def __init__(self,alpha = 1.0):
        self.alpha = alpha
        self.nlp = spacy.load("en_core_web_sm")
        self.Class_Probability = [0,0,0,0,0,0]
        self.Class = {
            "fear" : 0,
            "sad" : 1,
            "anger" : 2,
            "love" : 3,
            "suprise" : 4,
            "joy" :5
        }
        self.Vocabulary = {}
        self.StopWords = ["i", "","me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while","of", "at", "by", "for", "with", 
            "about","between", "into", "through", "during", "before", "after", "to", "from", "up", "down", "in", "out", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "nor", "only", "own", "same", "so",
            "than", "can", "will", "just", "now","do"]


    def Train(self,Data):
        for ind,line in Data.iterrows():
            words = self.Tokeniser(line[0])
            class_index = self.Class[line[1]]
            for word in words:
                if word not in self.Vocabulary:
                    self.Vocabulary[word] = [0,0,0,0,0,0]
                
                (self.Vocabulary[word])[class_index]+=1
                self.Class_Probability[class_index]+=1

        V = len(self.Vocabulary)

        for i in range(0,6):
            for word in self.Vocabulary:
                (self.Vocabulary[word])[i] = (((self.Vocabulary[word])[i] + self.alpha)/(self.Class_Probability[i] + (V*(self.alpha))))

        tmp = sum(self.Class_Probability)
        for i in range(0,6):
            self.Class_Probability[i] = (self.Class_Probability[i])/tmp


    def Test(self,TestData,max_count):
        score = 0
        for ind,text in TestData.iterrows():
            words = self.Tokeniser(text[0])

            Scores = [0]*6
            for i in range(0,6):
                Scores[i] = math.log(self.Class_Probability[i])

            for word in words:
                if word in self.Vocabulary:
                    word_probability = self.Vocabulary[word]
                    for i in range(0,6):
                        Scores[i] += (math.log(word_probability[i]))

    
            if self.Class[text[1]] == Scores.index(max(Scores)):
                score+=1

        return score
    

    def Predict(self,text):
        words = self.Tokeniser(text)
        Score = [0,0,0,0,0,0]
        for i in range(0,6):
            Score[i] = self.Class_Probability[i]

        for word in words:
            if word in self.Vocabulary:
                for i in range(0,6):
                    Score[i] += math.log((self.Vocabulary[word])[i])

        return Score
            
                       
    def lemmatization(self,word):
        return self.nlp(word)[0].lemma_

    def Tokeniser(self,text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = list(set(text.split(" ")))
        text = [self.lemmatization(tex) for tex in text if tex not in self.StopWords]
        return text



