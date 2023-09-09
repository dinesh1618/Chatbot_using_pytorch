import nltk
import numpy as np
class TextProcessing(object):
    def __init__(self, model):
        self.model = model
        if self.model.lower() == "nltk":
            import nltk
        if self.model.lower() == "spacy":
            import spacy
            self.nlp = spacy.load('en_core_web_sm')

    def SentTokenize(self, sent):
        if self.model == 'nltk':
            return nltk.word_tokenize(sent)
        else:
            return self.nlp(sent)

    def StopWordsRemoval(self, tokens, deep_remove=False, extra_tokens=None):
        if self.model == 'nltk':
            from nltk.corpus import stopwords
            if extra_tokens is not None:
                stwords = stopwords.words("english") + list(extra_tokens)
                return [word for word in tokens if word not in stwords]
            stwords = stopwords.words("english")
            return [word for word in tokens if word not in stwords]
        else:
            from spacy.lang.en.stop_words import STOP_WORDS
            if extra_tokens is not None:
                stwords = list(STOP_WORDS) + list(extra_tokens)
                return [word for word in tokens if word not in stwords]
            stwords = list(STOP_WORDS)
            return [word for word in tokens if word not in stwords]
        
    def Stemmimg(self, word):
        if self.model == 'nltk':
            from nltk.stem import PorterStemmer
            ps = PorterStemmer()
            return ps.stem(word.lower())
        else:
            return str(word.lemma_)
            
    
    def BogofWords(self, tokenized_sentence, words):
        sentence_words = [self.Stemmimg(word) for word in tokenized_sentence]
        bag = np.zeros(len(words), dtype=np.float32)
        for idx, w in enumerate(words):
            if w in sentence_words:
                bag[idx] = 1
        return bag