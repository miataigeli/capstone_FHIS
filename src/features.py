import os
import re
import json
import time
import spacy
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from urllib.request import urlopen


class feature_pipeline:
    """
    Pipeline for extracting features from a text in preparation for downstream
    classification tasks
    """

    def __init__(self, text="", flatten=True, class_mode="document"):
        assert class_mode.lower() in [
            "document",
            "sentence",
        ], "Classification can only be done at the document level or the sentence level! ('document' or 'sentence' allowed)"

        self.text = text
        self.flatten = flatten
        self.class_mode = class_mode
        self.nlp = spacy.load("es_core_news_md")
        self.freq_list = self.frequency_list_50k()

        self.sentences = []
        self.tokens = []
        self.lemmas = []
        self.pos_tags = []
        self.morphs = []
        self.parses = []
        if text:
            self.preprocess(self.text)
            self.spacy_pipeline(self.text)

    def flatten(self, list_of_sents):
        """utility function"""
        return [tok for sent in list_of_sents for tok in sent]

    def preprocess(self, text=self.text):
        """
        Process the given text to remove trailing numbers and whitespaces

        text: (str) the text (story, poem, paragraph, chapter) to process

        return: (str) the processed text
        """
        text_processed = []
        for s in text.strip().split("\n"):
            if len(s):
                s = re.sub(r"\b\d{,3}\b", "", s.strip())
                s = re.sub(r"\s+", " ", s)
                text_processed.append(s.strip())
        text_processed = list(filter(lambda s: not s.isspace(), text_processed))
        self.text = " ".join(text_processed)
        return self.text

    def get_sentences(self, text=self.text):
        """"""
        self.sentences = []
        for sent in self.nlp(text).sents:
            self.sentences.append(sent.text)
        return self.sentences

    def get_tokens(self, text=self.text, flatten=self.flatten):
        """"""
        self.tokens = []
        if flatten:
            for token in self.nlp(text):
                self.tokens.append(token.text)
        else:
            for sent in self.nlp(text).sents:
                sent_tokens = []
                for token in sent:
                    sent_tokens.append(token.text)
                self.tokens.append(sent_tokens)
        return self.tokens

    def get_lemmas(self, text=self.text, flatten=self.flatten):
        """"""
        self.lemmas = []
        if flatten:
            for token in self.nlp(text):
                self.lemmas.append(token.lemma_)
        else:
            for sent in self.nlp(text).sents:
                sent_lemmas = []
                for token in sent:
                    sent_lemmas.append(token.lemma_)
                self.lemmas.append(sent_lemmas)
        return self.lemmas

    def get_pos_tags(self, text=self.text, flatten=self.flatten):
        """"""
        self.pos_tags = []
        if flatten:
            for token in self.nlp(text):
                self.pos_tags.append(token.pos_)
        else:
            for sent in self.nlp(text).sents:
                sent_tags = []
                for token in sent:
                    sent_tags.append(token.pos_)
                self.pos_tags.append(sent_tags)
        return self.pos_tags

    def get_morphology(self, text=self.text, flatten=self.flatten):
        """"""
        self.morphs = []
        if flatten:
            for token in self.nlp(text):
                self.morphs.append(token.tag_)
        else:
            for sent in self.nlp(text).sents:
                sent_morphs = []
                for token in sent:
                    sent_morphs.append(token.tag_)
                self.morphs.append(sent_morphs)
        return self.morphs

    def get_dependency_parses(self, text=self.text, flatten=self.flatten):
        """"""
        self.parses = []
        if flatten:
            for token in self.nlp(text):
                self.parses.append(token.dep_)
        else:
            for sent in self.nlp(text).sents:
                sent_parses = []
                for token in sent:
                    sent_parses.append(token.dep_)
                self.parses.append(sent_parses)
        return self.parses

    def full_spacy(self, text=self.text):
        """
        Run the given text through the pretrained spaCy pipeline to extract
        sentences, tokens, lemmas, POS tags, morphology, and dependency parses
        for each sentence in the text.

        text: (str) the text (story, poem, paragraph, chapter) to process

        (no return values, the processed items are saved to lists that are
        attributes of the text_processor object)
        """
        self.sentences = []
        self.tokens = []
        self.lemmas = []
        self.pos_tags = []
        self.morphs = []
        self.parses = []
        doc = self.nlp(text)
        for sent in doc.sents:
            sent_tokens = []
            sent_lemmas = []
            sent_tags = []
            sent_morphs = []
            sent_parses = []
            for token in sent:
                sent_tokens.append(token.text)
                sent_lemmas.append(token.lemma_)
                sent_tags.append(token.pos_)
                sent_morphs.append(token.tag_)
                sent_parses.append(token.dep_)
            self.sentences.append(sent.text)
            self.tokens.append(sent_tokens)
            self.lemmas.append(sent_lemmas)
            self.pos_tags.append(sent_tags)
            self.morphs.append(sent_morphs)
            self.parses.append(sent_parses)
            self.sents.append(sent.text)

    def frequency_list_50k(self, file="./es_50k.txt"):
        """
        Given a path to a text file of 50k Spanish words ordered by frequency of
        occurrence, return a list of the words in order.

        file: (str) the path of the frequency list text file

        return: (list) a list of Spanish words
        """
        freq_list = []
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                freq_list.append(line.strip().split(" ")[0])
        return freq_list

    def freq_lookup(self, word, freq_list=self.freq_list):
        """
        Given a word to look up in an ordered frequency list, return the position
        of that word in the list (1-indexed). If the word is not present return 0.

        word: (str) the word to search in the list
        freq_list: (list[str]) the ordered list of words to search through

        return: (int) the index of the word if present in the list
        """
        try:
            idx = freq_list.index(word) + 1
        except:
            idx = 0
        return idx

    def word_ranks(self, text, freq_list=self.freq_list):
        """
        Given a tokenized text in the form of a list of lists of tokens, and a
        frequency list to search through, return a list of lists of integers,
        where the integers correspond to the ranks of the words in a frequency list.

        text: (list[list[str]]) the tokenized text
        freq_list: (list[str]) the ordered list of words to search through

        return: (list[list[int]]) the ranks of each of the tokens in the text
        """
        ranked_tokens = []
        for sent in text:
            ranked_sent = []
            for token in sent:
                ranked_sent.append(freq_lookup(token, freq_list))
            ranked_tokens.append(ranked_sent)
        return ranked_tokens

    def num_tokens(self, token_list=self.tokens):
        """"""
        return len(token_list)
    
    def ttr(self, token_list=self.tokens):
        """"""
        types = set(token_list)
        return len(types)/len(token_list)
    
    def pos_proportions(self, pos_list=self.pos_tags):
        """"""
        pos_counts = Counter(pos_list)
        pos_props = {}
        for pos, count in pos_counts.items():
            pos_props[pos] = count/len(pos_list)
        return pos_props
    
    def feature_extractor(self, text=self.text):
        """"""
        self.preprocess(text)
        self.get_tokens(flatten=True)
        self.get_pos_tags(flatten=True)
        num_tokens = self.num_tokens()
        ttr = self.ttr()
        pos_props = self.pos_proportions()
        if self.class_mode == "document":
            