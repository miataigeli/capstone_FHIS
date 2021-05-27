import os
import re
import json
import time
import pylabeador
import spacy
from spacy.lang.es.stop_words import STOP_WORDS
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import pandas as pd
from urllib.request import urlopen
from statistics import mean
from utils import A1


class feature_pipeline:
    """
    Pipeline for extracting features from texts in preparation for downstream
    classification tasks.
    """

    def __init__(
        self,
        text="",
        flatten=True,
        class_mode="document",
        freq_list_type="df",
        full_spacy=False,
    ):
        """
        Initialize object attritubtes from parameters and run pre-processing
        setup functions (eg., load spaCy pipeline, build frequency lists, etc.)

        text: (str) the text (story, poem, paragraph, chapter, etc.) to process
        flatten: (bool) flag to flatten lists of tokens/lemmas/etc.
        class_mode: (str) whether to generate features at the document level or
                    the sentence level
        freq_list_type: (str) whether the frequency list is a list object or a
                        pandas DataFrame object
        full_spacy: (bool) flag to specify whether to extract sentences, tokens,
                    lemmas, POS tags, morphology tags, and dependency parses of
                    a given text by default
        """

        assert class_mode.lower() in [
            "document",
            "sentence",
        ], "Classification can only be done at the document level or the sentence level! ('document' or 'sentence' allowed)"

        self.text = text
        self.flatten = flatten
        self.class_mode = class_mode
        self.nlp = spacy.load("es_core_news_md")
        if freq_list_type == "df":
            self.freq_list = self.frequency_list_10k()
            self.word_ranks = self.word_ranks_from_df
        else:
            self.freq_list = self.frequency_list_50k()
            self.word_ranks = self.word_ranks_from_list

        self.sentences = []
        self.tokens = []
        self.lemmas = []
        self.pos_tags = []
        self.morphs = []
        self.parses = []

        if text:
            _ = self.preprocess()
        if full_spacy:
            self.full_spacy()

    def flatten(self, list_of_sents):
        """
        Flatten a list of lists of tokens into just a list of tokens.

        list_of_sents: (list[list[str]]) a tokenized text arranged by sentences

        return: (list[str]) a list of tokens from the text
        """
        return [tok for sent in list_of_sents for tok in sent]

    def preprocess(self, text=None):
        """
        Process the given text to remove trailing numbers and whitespaces

        text: (str) the text (story, poem, paragraph, chapter) to process

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
                text_processed.append(s.strip().lower())
        text_processed = list(filter(lambda s: not s.isspace(), text_processed))
        self.text = " ".join(text_processed)
        return self.text

    def get_sentences(self, text=None):
        """
        Return the sentences from a raw text

        text: (str) a raw text

        return: (list[str]) list of sentences in the text
        """
        if text is None:
            text = self.text

        self.sentences = []
        for sent in self.nlp(text).sents:
            self.sentences.append(sent.text)
        return self.sentences

    def get_tokens(self, text=None):
        """
        Return the tokens from a raw text. If the attribute self.flatten is True
        the function returns a flat list of tokens, otherwise the tokens are
        arranged by sentences.

        text: (str) a raw text

        return: (list[str]) / (list[list[str]]) list of tokens in the text
        """
        if text is None:
            text = self.text

        self.tokens = []
        if self.flatten:
            for token in self.nlp(text):
                self.tokens.append(token.text)
        else:
            for sent in self.nlp(text).sents:
                sent_tokens = []
                for token in sent:
                    sent_tokens.append(token.text)
                self.tokens.append(sent_tokens)
        return self.tokens

    def get_lemmas(self, text=None):
        """
        Return the lemmas from a raw text. If the attribute self.flatten is True
        the function returns a flat list of lemmas, otherwise the lemmas are
        arranged by sentences.

        text: (str) a raw text

        return: (list[str]) / (list[list[str]]) list of lemmas in the text
        """
        if text is None:
            text = self.text

        self.lemmas = []
        if self.flatten:
            for token in self.nlp(text):
                self.lemmas.append(token.lemma_)
        else:
            for sent in self.nlp(text).sents:
                sent_lemmas = []
                for token in sent:
                    sent_lemmas.append(token.lemma_)
                self.lemmas.append(sent_lemmas)
        return self.lemmas

    def get_pos_tags(self, text=None):
        """
        Return the POS tags from a raw text. If the attribute self.flatten is True
        the function returns a flat list of tags, otherwise the tags are
        arranged by sentences.

        text: (str) a raw text

        return: (list[str]) / (list[list[str]]) list of POS tags in the text
        """
        if text is None:
            text = self.text

        self.pos_tags = []
        if self.flatten:
            for token in self.nlp(text):
                self.pos_tags.append(token.pos_)
        else:
            for sent in self.nlp(text).sents:
                sent_tags = []
                for token in sent:
                    sent_tags.append(token.pos_)
                self.pos_tags.append(sent_tags)
        return self.pos_tags

    def get_morphology(self, text=None):
        """
        Return the morphologized tags from a raw text. If the attribute
        self.flatten is True the function returns a flat list of morphologized
        tags, otherwise the morphologized tags are arranged by sentences.

        text: (str) a raw text

        return: (list[str]) / (list[list[str]]) list of tags in the text
        """
        if text is None:
            text = self.text

        self.morphs = []
        if self.flatten:
            for token in self.nlp(text):
                self.morphs.append(token.tag_)
        else:
            for sent in self.nlp(text).sents:
                sent_morphs = []
                for token in sent:
                    sent_morphs.append(token.tag_)
                self.morphs.append(sent_morphs)
        return self.morphs

    def get_dependency_parses(self, text=None):
        """
        Return the dependency parses from a raw text. If the attribute
        self.flatten is True the function returns a flat list of dependency
        parses, otherwise the dependency parses are arranged by sentences.

        text: (str) a raw text

        return: (list[str]) / (list[list[str]]) list of parses in the text
        """
        if text is None:
            text = self.text

        self.parses = []
        if self.flatten:
            for token in self.nlp(text):
                self.parses.append(token.dep_)
        else:
            for sent in self.nlp(text).sents:
                sent_parses = []
                for token in sent:
                    sent_parses.append(token.dep_)
                self.parses.append(sent_parses)
        return self.parses

    def full_spacy(self, text=None):
        """
        Run the given text through the pretrained spaCy pipeline to extract
        sentences, tokens, lemmas, POS tags, morphology, and dependency parses
        for each sentence in the text.

        text: (str) the text (story, poem, paragraph, chapter, etc.) to process

        (no return values, the processed items are saved to lists that are
        attributes of the text_processor object)
        """
        if text is None:
            text = self.text

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

    def freq_lookup_from_list(self, word, freq_list=None):
        """
        Given a word to look up in an ordered frequency list, return the position
        of that word in the list (1-indexed). If the word is not present return 0.

        word: (str) the word to search in the list
        freq_list: (list[str]) the ordered list of words to search through

        return: (int) the index of the word if present in the list
        """
        if freq_list is None:
            freq_list = self.freq_list

        try:
            idx = freq_list.index(word) + 1
        except:
            idx = 0
        return idx

    def word_ranks_from_list(self, token_list=None, freq_list=None):
        """
        Given a tokenized text in the form of a lists of tokens and a frequency
        list to search through, return a list of integers, where the integers
        correspond to the ranks of the words in a frequency list.

        token_list: (list[str]) the tokenized text
        freq_list: (list[str]) the ordered list of words to search through

        return: (list[int]) the ranks of each of the tokens in the text
        """
        if token_list is None:
            token_list = self.tokens

        if freq_list is None:
            freq_list = self.freq_list

        assert isinstance(
            freq_list, list
        ), "Wrong data type for the frequency list! Please use list data type."

        ranked_tokens = []
        for token in token_list:
            ranked_tokens.append(freq_lookup_from_list(token, freq_list))
        return ranked_tokens

    def word_ranks_from_df(self, lemma_list=None, df=None):
        """
        Given a lemmatized text in the form of a lists of lemmas and a DataFrame
        frequency list to search through, return a list of integers, where the
        integers correspond to the ranks of the words in a frequency list.

        lemma_list: (list[str]) the lemmatized text
        df: (pandas.DataFrame) DataFrame containing the lemmas to search through

        return: (list[int]) the ranks of each of the lemmatized tokens
        """
        if lemma_list is None:
            lemma_list = self.lemmas

        if df is None:
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

    def num_tokens(self, token_list=None):
        """
        Return the number of tokens in a list of tokens (tokenized text)

        token_list: (list[str]) the tokenized text

        return: (int) the number of tokens
        """
        if token_list is None:
            token_list = self.tokens

        return len(token_list)

    def avg_sent_length(self, sentences=None):
        """
        Return the average number of tokens per sentence in a text

        sentences: (list[str]) a text split into a list of sentences

        return: (float) the average number of tokens per sentence
        """
        if sentences is None:
            sentences = self.sentences

        tokenizer = self.nlp.tokenizer
        return mean([len(tokenizer(sent)) for sent in sentences])

    def ttr(self, token_list=None):
        """
        Return the type-token ratio (TTR) for a text given a list of tokens

        token_list: (list[str]) the tokenized text

        return: (float) the type-token ratio
        """
        if token_list is None:
            token_list = self.tokens

        types = set(token_list)
        return len(types) / len(token_list)

    def pos_proportions(self, pos_list=None):
        """
        Given the list of POS tags of a text, extract the proportions of each
        POS tag in the text as well as the proportions of content words and
        function words in the text.
        
        pos_list: (list[str]) the POS tags of the text

        return:
            pos_props: (dict{float}) dict of proportions of POS tags in the text
            cat_props: (dict{float}) dict of proportions of content and function
                       words in the text
        """

        if pos_list is None:
            pos_list = self.pos_tags

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

    def pronoun_density(self, pos_list=None):
        """
        Return the density of pronouns in a text, calculated as the proportion
        of the number of pronouns and the number of non-pronouns.

        pos_list: (list[str]) the POS tags of the text

        return: (float) the pronoun density of the text
        """
        if pos_list is None:
            pos_list = self.pos_tags

        total_prons = 0
        for pos in pos_list:
            if pos == "PRON":
                total_prons += 1
        return total_prons / (len(pos_list) - total_prons)

    def logical_operators(self, token_list=None):
        """
        Return the density of logical operators in a text, calculated as the
        proportion of the number of logical operators and the number of non-operators.

        token_list: (list[str]) the tokenized text

        return: (float) the logical operator density of the text
        """
        if token_list is None:
            token_list = self.tokens

        LOGICAL_OPS = {"si", "y", "o", "u", "no"}  # if, and, or, not

        total_logical_ops = 0
        for token in token_list:
            if token in LOGICAL_OPS:
                total_logical_ops += 1
        return total_logical_ops / (len(token_list) - total_logical_ops)

    def connectives(self, text=None):
        """
        Return the number of connective phrases in a text.

        text: (str) the raw text

        return: (int) the number of connectives in the text
        """
        if text is None:
            text = self.text

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

    def a_level_vocab_features(self, token_list=None, pos_list=None):
        """
        Given a tokenized text in the form of a lists of tokens and the
        corresponding list of POS tags, extract the percentage of A-level word
        tokens and types after removing whitespaces and stopwords.

        token_list: (list[str]) the tokenized text
        pos_list: (list[str]) the POS tags of the tokenized text

        return:
            no_sw_length: (int) length of the text with stopwords removed
            a_vocab_percent: (float) percentage of A-level word tokens in the text
            a_vocab_percent_set: (float) percentage of A-level word types in the text
        """
        if token_list is None:
            token_list = self.tokens

        if pos_list is None:
            pos_list = self.pos_tags

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

    def fernandez_huerta_score(self, token_list=None, sentences=None):
        """
        This function calculates Fernandez-Huerta score (Spanish equivalent of
        Flesch score) and the average number of syllables per sentence for a
        text given as a list of sentences and a list of tokens.

        Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5831059/#:~:text=The%20Fernandez%2DHuerta%20Formula%20(Fern%C3%A1ndez,formulae%20(Flesch%2C%201948).&text=The%20output%20is%20an%20index,representing%20a%20more%20difficult%20text

        token_list: (list[str]) the tokenized text
        sentences: (list[str]) the list of sentences in the text

        fh_score: (float) the calculated Fernandez-Huerta score
        syls_per_sent: (float) the average number of syllables per sentence
        """
        if token_list is None:
            token_list = self.tokens

        if sentences is None:
            sentences = self.sentences

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

    def feature_extractor(self, text=None):
        """"""
        if text is None:
            text = self.text
        else:
            _ = self.preprocess(text)

        if self.class_mode == "document":
            _ = self.get_sentences()
            _ = self.get_tokens()
            _ = self.get_lemmas()
            _ = self.get_pos_tags()

        num_tokens = self.num_tokens()
        avg_sent_length = self.avg_sent_length()
        no_sw_len, a_level_token_prop, a_level_type_prop = self.a_level_vocab_features()
        num_connectives = self.connectives()
        logical_operator_density = self.logical_operators()
        pronoun_density = self.pronoun_density()
        ttr = self.ttr()
        avg_word_freq = self.avg_word_freq()
        fh_score, syls_per_sent = self.fernandez_huerta_score()
        pos_props, cat_props = self.pos_proportions()
        pos_props_labels = list(pos_props.keys())
        pos_props = list(pos_props.values())
        cat_props_labels = list(cat_props.keys())
        cat_props = list(cat_props.values())

        features = [
            num_tokens,
            no_sw_len,
            avg_sent_length,
            a_level_token_prop,
            a_level_type_prop,
            num_connectives,
            logical_operator_density,
            pronoun_density,
            ttr,
            avg_word_freq,
            fh_score,
            syls_per_sent,
        ]
        features.extend(pos_props)
        features.extend(cat_props)

        labels = [
            "total_tokens",
            "total_tokens_w/o_stopwords",
            "avg_sent_length",
            "proportion_of_a_level_tokens",
            "proportion_of_a_level_types",
            "num_connectives",
            "logical_operator_density",
            "pronoun_density",
            "ttr",
            "avg_rank_of_lemmas_in_freq_list",
            "fernandez_huerta_score",
            "syllables_per_sentence",
        ]
        labels.extend(pos_props_labels)
        labels.extend(cat_props_labels)

        return features, labels
