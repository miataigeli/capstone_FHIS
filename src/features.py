"""
NOTE:
Feature extraction at the sentence level is not consistent yet. Only document
level feature extraction works, and feature extraction functions are currently
only configured to work for fully flattened lists.
"""


import os
import re
import json
import time
import pylabeador
import spacy
import nltk
from nltk.corpus.reader.wordnet import WordNetCorpusReader
from spacy.lang.es.stop_words import STOP_WORDS
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen
from pathlib import Path
from statistics import mean
from stanza.server import CoreNLPClient
from treelib import Node, Tree
from utils import A1


class feature_pipeline:
    """
    Pipeline for extracting features from texts in preparation for downstream
    classification tasks.
    """

    def __init__(
        self,
        text="",
        flat=True,
        class_mode="document",
        freq_list_type="df",
        full_spacy=False,
        dep_parse_flag=False,
        dep_parse_classpath="",
    ):
        """
        Initialize object attritubtes from parameters and run pre-processing
        setup functions (eg., load spaCy pipeline, build frequency lists, etc.)

        text: (str) the text (story, poem, paragraph, chapter, etc.) to process
        flat: (bool) flag to flatten lists of tokens/lemmas/etc.
        class_mode: (str) whether to generate features at the document level or
                    the sentence level
        freq_list_type: (str) whether the frequency list is a list object or a
                        pandas DataFrame object
        full_spacy: (bool) flag to specify whether to extract sentences, tokens,
                    lemmas, POS tags, morphology tags, and dependency parses of
                    a given text by default
        dep_parse_flag: (bool) flag to specify whether to perform dependency tree
                        parsing using CoreNLP or not (False by default)
        dep_parse_classpath: (str) if dependency parsing using CoreNLP is to be
                             done, a path to the stanza_corenlp directory on the
                             system must be provided
        """

        assert class_mode.lower() in [
            "document",
            "sentence",
        ], "Classification can only be done at the document level or the sentence level! ('document' or 'sentence' allowed)"

        self.text = text
        self.flat = flat
        self.class_mode = class_mode
        self.dep_parse_flag = dep_parse_flag

        if self.dep_parse_flag:
            assert (
                dep_parse_classpath != ""
            ), "dep_parse_classpath must be explicitly specified!"
            
            dep_parse_classpath = Path(dep_parse_classpath)
            assert os.path.exists(
                dep_parse_classpath
            ), "the specified dep_parse_classpath does not exist on your system!"

            self.corenlp_client = CoreNLPClient(
                properties="es",
                classpath=dep_parse_classpath,
                annotators=["depparse"],
                timeout=30000,
                memory="5G",
            )

        self.nlp = spacy.load("es_core_news_md")
        self.wncr = None
        self.freq_list = None
        if freq_list_type == "df":
            self.word_ranks = self.word_ranks_from_df
        else:
            self.word_ranks = self.word_ranks_from_list

        self.sentences = []
        self.tokens = []
        self.lemmas = []
        self.pos_tags = []
        self.morphs = []
        self.parses = []
        self.noun_chunks = []

        if text:
            _ = self.preprocess()
        if full_spacy:
            self.full_spacy()

    def flatten(self, list_of_sents):
        """
        Utility Function:
        Flatten a list of lists of tokens into just a list of tokens.

        list_of_sents: (list[list[str]]) a tokenized text arranged by sentences

        return: (list[str]) a list of tokens from the text
        """
        return [tok for sent in list_of_sents for tok in sent]

    def preprocess(self, text=None):
        """
        Process the given text to remove trailing numbers and whitespaces

        text: (str) the text (story, poem, paragraph, chapter, etc.) to process

        return: (str) the processed text
        """
        if text is None:
            text = self.text

        text_processed = []
        for s in text.strip().split("\n"):
            if len(s):
                # Remove chapters/titles
                s = re.sub(r"[0-9]+\.[^\n]+\n", "", s.strip())
                # Remove stray numbers that are 3 digits or less
                s = re.sub(r"\b\d{,3}\b", "", s.strip())
                # Replace excessive whitespaces with single spaces
                s = re.sub(r"\s+", " ", s.strip())
                # Remove leftover leading periods and commas
                s = re.sub(r"^[,.]+", "", s.strip())
                text_processed.append(s.strip().lower())
        text_processed = list(filter(lambda s: not s.isspace(), text_processed))
        self.text = " ".join(text_processed)
        return self.text

    def get_sentences(self, text=None):
        """
        Return the sentences from a raw text

        text: (str) an unprocessed text (optional)

        return: (list[str]) list of sentences in the text
        """
        text = self.preprocess(text) if text else self.text

        self.sentences = []
        self.sentences = [sent.text for sent in self.nlp(text).sents]
        return self.sentences

    def get_tokens(self, text=None):
        """
        Return the tokens from a raw text. If the attribute self.flat is True
        the function returns a flat list of tokens, otherwise the tokens are
        arranged by sentences.

        text: (str) an unprocessed text (optional)

        return: (list[str]) / (list[list[str]]) list of tokens in the text
        """
        text = self.preprocess(text) if text else self.text

        self.tokens = []
        doc = self.nlp(text)
        if self.flat:
            self.tokens = [token.text for token in doc]
        else:
            for sent in doc.sents:
                self.tokens.append([token.text for token in sent])
        return self.tokens

    def get_lemmas(self, text=None):
        """
        Return the lemmas from a raw text. If the attribute self.flat is True
        the function returns a flat list of lemmas, otherwise the lemmas are
        arranged by sentences.

        text: (str) an unprocessed text (optional)

        return: (list[str]) / (list[list[str]]) list of lemmas in the text
        """
        text = self.preprocess(text) if text else self.text

        self.lemmas = []
        doc = self.nlp(text)
        if self.flat:
            self.lemmas = [token.lemma_ for token in doc]
        else:
            for sent in doc.sents:
                self.lemmas.append([token.lemma_ for token in sent])
        return self.lemmas

    def get_pos_tags(self, text=None):
        """
        Return the POS tags from a raw text. If the attribute self.flat is True
        the function returns a flat list of tags, otherwise the tags are
        arranged by sentences.

        text: (str) an unprocessed text (optional)

        return: (list[str]) / (list[list[str]]) list of POS tags in the text
        """
        text = self.preprocess(text) if text else self.text

        self.pos_tags = []
        doc = self.nlp(text)
        if self.flat:
            self.pos_tags = [token.pos_ for token in doc]
        else:
            for sent in doc.sents:
                self.pos_tags.append([token.pos_ for token in sent])
        return self.pos_tags

    def get_morphology(self, text=None):
        """
        Return the morphologized tags from a raw text. If the attribute
        self.flat is True the function returns a flat list of morphologized
        tags, otherwise the morphologized tags are arranged by sentences.

        text: (str) an unprocessed text (optional)

        return: (list[str]) / (list[list[str]]) list of tags in the text
        """
        text = self.preprocess(text) if text else self.text

        self.morphs = []
        doc = self.nlp(text)
        if self.flat:
            self.morphs = [token.tag_ for token in doc]
        else:
            for sent in doc.sents:
                self.morphs.append([token.tag_ for token in sent])
        return self.morphs

    def get_dependency_parses(self, text=None):
        """
        Return the dependency parses from a raw text. If the attribute
        self.flat is True the function returns a flat list of dependency
        parses, otherwise the dependency parses are arranged by sentences.

        text: (str) an unprocessed text (optional)

        return: (list[str]) / (list[list[str]]) list of parses in the text
        """
        text = self.preprocess(text) if text else self.text

        self.parses = []
        doc = self.nlp(text)
        if self.flat:
            self.parses = [token.dep_ for token in doc]
        else:
            for sent in doc.sents:
                self.parses.append([token.dep_ for token in sent])
        return self.parses

    def get_noun_chunks(self, text=None):
        """
        Returns noun phrases from a raw text. If the attribute self.flat is True
        the function returns a flat list of noun chunks, otherwise the noun
        chunks are arranged by sentences.

        text: (str) an unprocessed text (optional)

        return: (list[str] / list[list[str]]) list of noun chunks in the text
        """
        text = self.preprocess(text) if text else self.text

        self.noun_chunks = []
        doc = self.nlp(text)
        if self.flat:
            self.noun_chunks = [chunk.text for chunk in doc.noun_chunks]
        else:
            for sent in doc.sents:
                sent_doc = self.nlp(sent.text)
                sent_npc = [chunk.text for chunk in sent_doc.noun_chunks]
                self.noun_chunks.append(sent_npc)
        return self.noun_chunks

    def full_spacy(self, text=None):
        """
        !!! NOTE: Only run this method if you absolutely NEED to extract all
        of the spaCy features. It is typically more efficient to instead run
        only the `get_` methods of those spaCy features that you require. !!!

        Run the given text through the pretrained spaCy pipeline to extract
        sentences, tokens, lemmas, POS tags, morphology, and dependency parses
        for each sentence in the text.

        text: (str) the text (story, poem, paragraph, chapter, etc.) to process

        (no return values, the processed items are saved to lists that are
        attributes of the text_processor object)
        """
        text = self.preprocess(text) if text else self.text

        self.sentences = []
        self.tokens = []
        self.lemmas = []
        self.pos_tags = []
        self.morphs = []
        self.parses = []
        self.noun_chunks = []
        self.get_noun_chunks()
        doc = self.nlp(text)
        if self.flat:
            for token in doc:
                self.tokens.append(token.text)
                self.lemmas.append(token.lemma_)
                self.pos_tags.append(token.pos_)
                self.morphs.append(token.tag_)
                self.parses.append(token.dep_)
            for sent in doc.sents:
                self.sentences.append(sent.text)
        else:
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
                self.sentences.append(sent.text)

    def frequency_list_10k(self):
        """
        Extract words, their frequencies and their lemmatized forms from Wiktionary
        frequency lists of the top 10,000 words from Spanish subtitling data.

        return: (pandas.DataFrame) a DataFrame of Spanish words, their
                frequencies and their lemmatized forms
        """

        urls = [
            "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish1000",
            "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish1001-2000",
            "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish2001-3000",
            "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish3001-4000",
            "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish4001-5000",
            "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish5001-6000",
            "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish6001-7000",
            "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish7001-8000",
            "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish8001-9000",
            "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish9001-10000",
        ]

        df_list = []
        for url in urls:
            soup = BeautifulSoup(urlopen(url), "lxml")
            table = soup.find("tbody")
            for row in table.find_all("tr")[1:]:
                row = row.text.lower().strip().split("\n")
                df_list.append(list(filter(lambda e: e != "", row))[1:])
        df = pd.DataFrame(df_list, columns=["word", "occurrences", "lemma"])

        return df

    def frequency_list_50k(self, file="../vocab/es_50k.txt"):
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

    def freq_lookup_from_list(self, word, freq_list):
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

    def word_ranks_from_list(self, text=None, token_list=None, freq_list=None):
        """
        Given an unprocessed text, or a tokenized text in the form of a list of
        tokens, and a frequency list to search through, return a list of integers
        where the integers correspond to the ranks of the words in a frequency list.

        text: (str) an unprocessed text (not necessary if token_list is given)
        token_list: (list[str]) the tokenized text (not necessary if text is given)
        freq_list: (list[str]) the ordered list of words to search through

        return: (list[int]) the ranks of each of the tokens in the text
        """
        if text:
            text = self.preprocess(text)
            token_list = self.get_tokens()

        token_list = self.tokens if self.tokens else self.get_tokens()

        if freq_list is None:
            if self.freq_list is None:
                self.freq_list = self.frequency_list_10k()
            freq_list = self.freq_list

        assert isinstance(
            freq_list, list
        ), "Wrong data type for the frequency list! Please use list data type."

        ranked_tokens = []
        for token in token_list:
            ranked_tokens.append(freq_lookup_from_list(token, freq_list))
        return ranked_tokens

    def word_ranks_from_df(self, text=None, lemma_list=None, df=None):
        """
        Given an unprocessed text, or a lemmatized text in the form of a list of
        lemmas, and a DataFrame frequency list to search through, return a list
        of integers, where the integers correspond to the ranks of the words in
        a frequency list.

        text: (str) an unprocessed text (not necessary if lemma_list is given)
        lemma_list: (list[str]) the lemmatized text (not necessary if text is given)
        df: (pandas.DataFrame) DataFrame containing the lemmas to search through

        return: (list[int]) the ranks of each of the lemmatized tokens
        """
        if text:
            text = self.preprocess(text)
            lemma_list = self.get_lemmas()

        lemma_list = self.lemmas if self.lemmas else self.get_lemmas()

        if df is None:
            if self.freq_list is None:
                self.freq_list = self.frequency_list_10k()
            df = self.freq_list

        assert isinstance(
            df, pd.DataFrame
        ), "Wrong data type for the frequency list! Please use Pandas DataFrame."

        ranked_text = []
        index = df.index
        for lemma in lemma_list:
            condition = df["lemma"] == lemma
            rank = index[condition].tolist()
            rank = rank[0] + 1 if rank else 0
            ranked_text.append(rank)
        return ranked_text

    def avg_word_freq(self):
        """
        Compute the average rank of a token in a text from a list of Spanish
        token frequencies.

        return: (float) the average token frequency rank
        """
        ranked_text = self.word_ranks()
        return sum(ranked_text) / len(ranked_text)

    def num_tokens(self, text=None, token_list=None):
        """
        Return the number of tokens in a list of tokens (tokenized text)

        text: (str) an unprocessed text (not necessary if token_list is given)
        token_list: (list[str]) the tokenized text (not necessary if text is given)

        return: (int) the number of tokens
        """
        if text:
            text = self.preprocess(text)
            token_list = self.get_tokens()

        token_list = self.tokens if self.tokens else self.get_tokens()

        return len(token_list)

    def avg_sent_length(self, text=None, sentences=None):
        """
        Return the average number of tokens per sentence in a text

        text: (str) an unprocessed text (not necessary if sentences is given)
        sentences: (list[str]) a text split into a list of sentences (not necessary if text is given)

        return: (float) the average number of tokens per sentence
        """
        if text:
            text = self.preprocess(text)
            sentences = self.get_sentences()

        sentences = self.sentences if self.sentences else self.get_sentences()

        tokenizer = self.nlp.tokenizer
        return mean([len(tokenizer(sent)) for sent in sentences])

    def ttr(self, text=None, token_list=None):
        """
        Return the type-token ratio (TTR) for a text given a list of tokens

        text: (str) an unprocessed text (not necessary if token_list is given)
        token_list: (list[str]) the tokenized text (not necessary if text is given)

        return: (float) the type-token ratio
        """
        if text:
            text = self.preprocess(text)
            token_list = self.get_tokens()

        token_list = self.tokens if self.tokens else self.get_tokens()

        types = set(token_list)
        return len(types) / len(token_list)

    def pos_proportions(self, text=None, pos_list=None):
        """
        Given the list of POS tags of a text, extract the proportions of each
        POS tag in the text as well as the proportions of content words and
        function words in the text.

        text: (str) an unprocessed text (not necessary if pos_list is given)
        pos_list: (list[str]) the POS tags of the text (not necessary if text is given)

        return:
            pos_props: (dict{float}) dict of proportions of POS tags in the text
            cat_props: (dict{float}) dict of proportions of content and function
                       words in the text
        """
        if text:
            text = self.preprocess(text)
            pos_list = self.get_pos_tags()

        pos_list = self.pos_tags if self.pos_tags else self.get_pos_tags()

        CONTENT_POS = {"VERB", "NOUN", "PROPN", "ADP", "ADJ", "ADV"}
        FUNCTION_POS = {
            "CONJ",
            "CCONJ",
            "SCONJ",
            "AUX",
            "DET",
            "PRON",
            "INTJ",
            "NUM",
            "PART",
        }

        cat_counts = {"CONTENT": 0, "FUNCTION": 0}

        pos_counts = {
            "ADJ": 0,
            "ADP": 0,
            "ADV": 0,
            "AUX": 0,
            "CONJ": 0,
            "CCONJ": 0,
            "DET": 0,
            "INTJ": 0,
            "NOUN": 0,
            "NUM": 0,
            "PART": 0,
            "PRON": 0,
            "PROPN": 0,
            "PUNCT": 0,
            "SCONJ": 0,
            "SYM": 0,
            "VERB": 0,
            "X": 0,
            "EOL": 0,
            "SPACE": 0,
        }

        total = 0
        for pos in pos_list:
            if pos in CONTENT_POS:
                cat_counts["CONTENT"] += 1
                total += 1
            elif pos in FUNCTION_POS:
                cat_counts["FUNCTION"] += 1
                total += 1
            pos_counts[pos] += 1

        pos_props = {}
        for pos, count in pos_counts.items():
            pos_props[pos] = count / len(pos_list)

        cat_props = {}
        for cat, count in cat_counts.items():
            cat_props[cat] = cat_counts[cat] / total

        return pos_props, cat_props

    def pronoun_density(self, text=None, pos_list=None):
        """
        Return the density of pronouns in a text, calculated as the proportion
        of the number of pronouns and the number of non-pronouns.

        text: (str) an unprocessed text (not necessary if pos_list is given)
        pos_list: (list[str]) the POS tags of the text (not necessary if text is given)

        return: (float) the pronoun density of the text
        """
        if text:
            text = self.preprocess(text)
            pos_list = self.get_pos_tags()

        pos_list = self.pos_tags if self.pos_tags else self.get_pos_tags()

        total_prons = 0
        for pos in pos_list:
            if pos == "PRON":
                total_prons += 1
        return total_prons / (len(pos_list) - total_prons)

    def logical_operators(self, text=None, token_list=None):
        """
        Return the density of logical operators in a text, calculated as the
        proportion of the number of logical operators and the number of non-operators.

        text: (str) an unprocessed text (not necessary if token_list is given)
        token_list: (list[str]) the tokenized text (not necessary if text is given)

        return: (float) the logical operator density of the text
        """
        if text:
            text = self.preprocess(text)
            token_list = self.get_tokens()

        token_list = self.tokens if self.tokens else self.get_tokens()

        LOGICAL_OPS = {"si", "y", "o", "u", "no"}  # if, and, or, not

        total_logical_ops = 0
        for token in token_list:
            if token in LOGICAL_OPS:
                total_logical_ops += 1
        return total_logical_ops / (len(token_list) - total_logical_ops)

    def connectives(self, text=None):
        """
        Return the number of connective phrases in a text.

        text: (str) a text

        return: (int) the number of connectives in the text
        """
        text = self.preprocess(text) if text else self.text

        CONNECTIVES = {
            "por eso",
            "a pesar de",
            "además",
            "y",
            "también",
            "incluso",
            "pero",
            "aunque",
            "sin embargo",
            "no obstante",
            "porque",
            "ya que",
            "puesto que",
            "debido a que",
            "a causa de que",
            "como",
            "así",
            "entonces",
            "por lo tanto",
            "en consecuencia",
            "después",
            "antes",
            "al mismo tiempo",
            "finalmente",
            "al principio",
            "por último",
            "dado que",
            "pese a",
            "es decir",
            "o sea",
            "y luego",
            "primero",
            "todavía",
            "aún",
            "cuando",
            "aunque",
            "por consiguiente",
            "consecuentemente",
            "por otra parte",
            "es decir",
            "por lo visto",
            "que yo sepa",
            "de todas formas",
            "de todas maneras",
            "aparte de",
            "tal como",
            "a vez de",
            "en concreto",
            "en pocas palabras",
            "tan pronto como",
            "mientras tanto",
            "hasta",
            "por último",
            "pues",
            "en cuanto",
            "por fin",
            "al mismo tiempo",
            "a la misma vez",
            "inmediatamente",
            "durante",
            "eventualmente",
            "frecuentemente",
            "al rato",
            "en primer lugar",
            "anoche",
            "luego",
            "nunca",
            "ahora",
            "muchas veces",
            "al otro día",
            "desde entonces",
            "raramente",
            "algunas veces",
            "pronto",
        }

        conn_count = 0
        for conn in CONNECTIVES:
            if conn in text:
                conn_count += 1

        return conn_count

    def a_level_vocab_features(self, text=None, token_list=None, pos_list=None):
        """
        Given a tokenized text in the form of a lists of tokens and the
        corresponding list of POS tags, extract the percentage of A-level word
        tokens and types after removing whitespaces and stopwords.

        text: (str) an unprocessed text (not necessary if token_list and pos_list are given)
        token_list: (list[str]) the tokenized text (not necessary if text is given)
        pos_list: (list[str]) the POS tags of the text (not necessary if text is given)

        return:
            no_sw_length: (int) length of the text with stopwords removed
            a_vocab_percent: (float) percentage of A-level word tokens in the text
            a_vocab_percent_set: (float) percentage of A-level word types in the text
        """
        if text:
            text = self.preprocess(text)
            token_list = self.get_tokens()
            pos_list = self.get_pos_tags()

        token_list = self.tokens if self.tokens else self.get_tokens()
        pos_list = self.pos_tags if self.pos_tags else self.get_pos_tags()

        a_vocab = A1().vocab
        no_sw_length = 0
        a_vocab_count = 0
        tok_set = set()

        for i in range(len(token_list)):
            tok = token_list[i]
            pos = pos_list[i]
            if (
                tok in STOP_WORDS
                or tok == " "
                or tok == "\n"
                or tok == "\xa0"
                or pos == "PUNCT"
            ):
                continue
            elif tok in a_vocab:
                a_vocab_count += 1
            tok_set.add(tok)
            no_sw_length += 1

        a_vocab_percent = a_vocab_count / no_sw_length
        a_vocab_percent_set = len(tok_set & a_vocab) / len(tok_set)

        return no_sw_length, a_vocab_percent, a_vocab_percent_set

    def fernandez_huerta_score(self, text=None, token_list=None, sentences=None):
        """
        This function calculates Fernandez-Huerta score (Spanish equivalent of
        Flesch score) and the average number of syllables per sentence for a
        text given as a list of sentences and a list of tokens.

        Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5831059/#:~:text=The%20Fernandez%2DHuerta%20Formula%20(Fern%C3%A1ndez,formulae%20(Flesch%2C%201948).&text=The%20output%20is%20an%20index,representing%20a%20more%20difficult%20text

        text: (str) an unprocessed text (not necessary if token_list and sentences are given)
        token_list: (list[str]) the tokenized text (not necessary if text is given)
        sentences: (list[str]) a text split into a list of sentences (not necessary if text is given)

        fh_score: (float) the calculated Fernandez-Huerta score
        syls_per_sent: (float) the average number of syllables per sentence
        """
        if text:
            text = self.preprocess(text)
            token_list = self.get_tokens()
            sentences = self.get_sentences()

        token_list = self.tokens if self.tokens else self.get_tokens()
        sentences = self.sentences if self.sentences else self.get_sentences()

        num_sents = len(sentences)
        num_tokens = len(token_list)
        # count as token only if the token contains at least one letter (eg., "Vd." is a token)
        num_alpha_tokens = len(
            [token for token in token_list if any(t.isalpha() for t in token)]
        )

        # if text contains nothing, set the score as very very easy to read
        if num_alpha_tokens == 0 or num_sents == 0:
            return 206

        num_syls = 0
        for token in token_list:
            # if the token contains at least one letter
            if any(t.isalpha() for t in token):
                try:
                    # get rid of non-alphabets in the token
                    token_ = "".join([t for t in token if t.isalpha()])
                    # and get syllables
                    num_syls += len(pylabeador.syllabify(token_))

                except:
                    # There are alphabets such as ª which cannot be processed
                    num_alpha_tokens -= 1

        fh_score = (
            206.835
            - 102 * (num_sents / num_alpha_tokens)
            - 60 * (num_syls / num_alpha_tokens)
        )

        syls_per_sent = num_syls / num_sents

        return fh_score, syls_per_sent

    def degree_of_abstraction(self, text=None, token_list=None, pos_list=None):
        """
        This function measures the degree of abstraction of a text by measuring
        the distance of its nouns to the top level in the wordnet.

        text: (str) an unprocessed text (not necessary if token_list and pos_list are given)
        token_list: (list[str]) the tokenized text (not necessary if text is given)
        pos_list: (list[str]) the POS tags of the text (not necessary if text is given)

        return:
            (float) the average degree of abstraction (the higher, less abstract),
            (float) the min degree of abstraction in the text
        """
        if text:
            text = self.preprocess(text)
            token_list = self.get_tokens()
            pos_list = self.get_pos_tags()

        token_list = self.tokens if self.tokens else self.get_tokens()
        pos_list = self.pos_tags if self.pos_tags else self.get_pos_tags()

        """
        Initialize Spanish WordNet
        !!! Must have Spanish WordNet extracted into the given directory !!!
        Download and extract this file:
        https://github.com/pln-fing-udelar/wn-mcr-transform/blob/master/wordnet_spa.tar.gz
        """
        if not self.wncr:
            result_root = "../wordnet_spa/"
            self.wncr = WordNetCorpusReader(result_root, None)

        top_synset = self.wncr.synset("entidad.n.01")  # Top synset
        sent_nouns, sent_levels = [], []
        num_levels, num_nouns = 0.0, 0

        for i_token, token in enumerate(token_list):
            # calculate levels for each sense of the token
            token_levels, num_senses = 0.0, 0
            tag = pos_list[i_token]
            token = token.lower()
            synsets = self.wncr.synsets(token)
            if len(synsets) > 0 and tag == "NOUN":
                for synset in synsets:
                    if synset.name().split(".")[1] == "n":  # only process noun
                        try:
                            levels = 1 / synset.path_similarity(top_synset)
                            token_levels += levels
                            num_senses += 1
                        except:
                            pass
                if num_senses > 0:
                    num_nouns += 1
                    sent_nouns.append(token)
                    sent_levels.append(token_levels / num_senses)
                    # average level over the senses
                    num_levels += token_levels / num_senses

        if num_nouns == 0:
            return 1000, 1000  # no abstraction
        else:
            # first returns the average number of levels in the text,
            # second returns the minimum num of levels in the text
            return num_levels / num_nouns, min(sent_levels)

    def polysemy_ambiguation(self, text=None, token_list=None, pos_list=None):
        """
        This function measures degree of ambiguation of a text by counting the
        number of senses of each content word in the text.

        text: (str) an unprocessed text (not necessary if token_list and pos_list are given)
        token_list: (list[str]) the tokenized text (not necessary if text is given)
        pos_list: (list[str]) the POS tags of the text (not necessary if text is given)

        return:
            (float) the mean degree of ambiguation over all tokens (the higher, more ambiguous),
            (float) the mean degree of ambiguation over all content tokens
        """
        if text:
            text = self.preprocess(text)
            token_list = self.get_tokens()
            pos_list = self.get_pos_tags()

        token_list = self.tokens if self.tokens else self.get_tokens()
        pos_list = self.pos_tags if self.pos_tags else self.get_pos_tags()

        """
        Initialize Spanish WordNet
        !!! Must have Spanish WordNet extracted into the given directory !!!
        Download and extract this file:
        https://github.com/pln-fing-udelar/wn-mcr-transform/blob/master/wordnet_spa.tar.gz
        """
        if not self.wncr:
            result_root = "../wordnet_spa/"
            self.wncr = WordNetCorpusReader(result_root, None)

        sent_senses = []
        sent_cont_tokens, sent_cont_senses = [], []
        num_senses, num_cont_senses = 0, 0

        CONTENT_POS = {"VERB", "NOUN", "PROPN", "ADP", "ADJ", "ADV"}

        for i_token, token in enumerate(token_list):
            token = token.lower()
            synsets = self.wncr.synsets(token)
            # All words
            if len(synsets) > 0:
                num_senses += len(synsets)
                sent_senses.append(len(synsets))
            # Only content words
            if pos_list[i_token] in CONTENT_POS and len(synsets) > 0:
                num_cont_senses += len(synsets)
                sent_cont_tokens.append(token)
                sent_cont_senses.append(len(synsets))

        if sent_senses == []:
            return 0, 0

        return mean(sent_senses), mean(sent_cont_senses)

    def density_noun_chunks(self, text=None, noun_chunks=None):
        """
        This function calculates the mean number of modifiers of noun phrases in
        the given text (either a text string or a list of noun phrase chunks).

        text: (str) an unprocessed text (not necessary if noun_chunks is given)
        noun_chunks: (list[str]) the noun phrase chunks in the text (not necessary if text is given)

        return: (float) the mean number of noun phrases modifiers in the text
        """
        if text:
            text = self.preprocess(text)
            noun_chunks = self.get_noun_chunks()

        noun_chunks = self.noun_chunks if self.noun_chunks else self.get_noun_chunks()

        try:
            len_noun_chunks = 0
            if len(noun_chunks) > 0:
                len_noun_chunks = (
                    mean([len(chunk.strip().split()) for chunk in noun_chunks]) - 1
                )
            else:
                len_noun_chunks = 0

            return len_noun_chunks

        except:
            return 0

    def dependency_tree(self, sent):
        """
        This function returns the depth of sentence using dependency parsing.
        Assumptions:
         1. The input sent is a CoreNLP_pb2.Sentence data structure.
         2. The dependency parsing information in sent (from CoreNLP) is correct.
            (hence, dependency parsing is out of scope for testing)
        Note: If sent has no edges, this function returns an empty tree.
              Therefore, a one-word sentence will return an empty tree.

        sent: (CoreNLP_pb2.Sentence) sentence for which the dependency tree is built

        return: (int) depth of the tree
        """
        tree_list = [Tree()]

        for edge in sent.basicDependencies.edge:
            source = edge.source  # source of the edge
            target = edge.target  # target of the edge

            source_tree_idx = -1
            target_tree_idx = -1
            # find a tree which contains source and call it source_tree
            for i, tree in enumerate(tree_list):
                # find a tree which contains target and call it target_tree
                if tree.get_node(source):
                    source_tree_idx = i
                if tree.get_node(target):
                    target_tree_idx = i

            # if no source tree, no target tree; then, create a new tree
            if source_tree_idx < 0 and target_tree_idx < 0:
                new_tree = Tree()
                new_tree.create_node(source, source)
                new_tree.create_node(target, target, parent=source)
                tree_list.append(new_tree)

            # if source_tree exists and no target_tree, add a target node
            elif target_tree_idx < 0:
                tree = tree_list[source_tree_idx]
                tree.create_node(target, target, parent=source)

            # if target_tree exists and no source_tree,
            #  add the source node as the root of the tree
            elif source_tree_idx < 0:
                new_tree = Tree()
                new_tree.create_node(source, source)
                sub_tree = tree_list[target_tree_idx]
                new_tree.paste(source, sub_tree)
                tree_list[target_tree_idx] = new_tree

            # if both source_tree and target_tree exist, connect these trees
            else:
                assert source_tree_idx != target_tree_idx
                source_tree = tree_list[source_tree_idx]
                target_tree = tree_list[target_tree_idx]

                assert target_tree.root == target
                source_tree.paste(source, target_tree)
                tree_list[source_tree_idx] = source_tree
                tree_list[target_tree_idx] = Tree()

        # get the tree depth for each tree in the list
        tree_depth_list = [tree.depth() for tree in tree_list]

        # the tree with max depth is the final tree
        return max(tree_depth_list), tree_list[np.argmax(tree_depth_list)]

    def depth_dep_parse(self, text=None):
        """
        Given a text as a string this function calculates the average depth of
        the sentences in the text using the dependency parse trees from CoreNLP.
        This serves as a measure of the complexity of the sentences in the text.

        Assumes that CoreNLPClient is already running.

        text: (str) a text

        return: (float) the average depth of the sentences in the text
        """
        text = self.preprocess(text) if text else self.text

        spanish_ann = self.corenlp_client.annotate(text)

        avg_depth = 0
        for sent in spanish_ann.sentence:
            depth, _ = dependency_tree(sent)
            avg_depth += depth

        if len(spanish_ann.sentence) == 0:
            return 0
        else:
            return avg_depth / len(spanish_ann.sentence)

    def feature_extractor(self, text=None):
        """
        Perform preprocessing and extract all the features from the text

        text: (str) the text (story, poem, paragraph, chapter, etc.) to process

        return: (dict) the features extracted from the text
        """
        text = self.preprocess(text) if text else self.text

        if self.class_mode == "document":
            _ = self.get_sentences()
            _ = self.get_tokens()
            _ = self.get_lemmas()
            _ = self.get_pos_tags()
            _ = self.get_noun_chunks()

        num_tokens = self.num_tokens()
        avg_sent_length = self.avg_sent_length()
        no_sw_len, a_level_token_prop, a_level_type_prop = self.a_level_vocab_features()
        num_connectives = self.connectives()
        logical_operator_density = self.logical_operators()
        pronoun_density = self.pronoun_density()
        ttr = self.ttr()
        avg_word_freq = self.avg_word_freq()
        fh_score, syls_per_sent = self.fernandez_huerta_score()
        avg_abstraction, min_abstraction = self.degree_of_abstraction()
        avg_ambiguation, avg_content_ambiguation = self.polysemy_ambiguation()
        np_density = self.density_noun_chunks()
        if self.dep_parse_flag:
            avg_text_depth = self.depth_dep_parse()
        pos_props, cat_props = self.pos_proportions()

        features = {
            "total_tokens": num_tokens,
            "total_tokens_w/o_stopwords": no_sw_len,
            "avg_sent_length": avg_sent_length,
            "proportion_of_A_level_tokens": a_level_token_prop,
            "proportion_of_A_level_types": a_level_type_prop,
            "num_connectives": num_connectives,
            "logical_operator_density": logical_operator_density,
            "pronoun_density": pronoun_density,
            "type_token_ratio": ttr,
            "avg_rank_of_lemmas_in_freq_list": avg_word_freq,
            "fernandez_huerta_score": fh_score,
            "syllables_per_sentence": syls_per_sent,
            "avg_degree_of_abstraction": avg_abstraction,
            "min_degree_of_abstraction": min_abstraction,
            "avg_ambiguation_all_words": avg_ambiguation,
            "avg_ambiguation_content_words": avg_content_ambiguation,
            "noun_phrase_density": np_density,
        }
        if self.dep_parse_flag:
            features.update({"avg_dep_tree_depth": avg_text_depth})
        features.update(pos_props)
        features.update(cat_props)

        return features
