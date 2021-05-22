import os
import re
import json
import time
import spacy
from collections import defaultdict
from bs4 import BeautifulSoup
from urllib.request import urlopen


def read_corpus(path="../corpus/"):
    """
    Given a path to a directory containing JSON files of the scraped corpus
    documents and their metadata, load them all into a dict{list[dicts]}
    such that:
    {
        "A1": [{"source": "...", "content": "...", ...}, {...}],
        "A2": [...],
        ...
    }

    path: (str) the path of the directory containing the JSON files

    return: (dict{list[dicts]}) a dictionary of texts arranged by reading level
    (a text is a single cohesive piece of reading material, be it a short
    story, a poem, song lyrics, a book chapter, etc.)
    """

    corpus = defaultdict(list)
    for file in os.listdir(path):
        if "json" in file:
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                doc_list = json.load(f)
                for d in doc_list:
                    level = d["level"]
                    if level == "A2/B1":
                        level = "B1"
                    corpus[level].append(d)
    return corpus


def frequency_list_50k(file="./es_50k.txt"):
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


def freq_lookup(word, freq_list):
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


def word_ranks(text, freq_list):
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


class A1:
    """
    Class definition for loading, scraping, writing A1-level vocabularies
    """

    def __init__(self, mode="txt"):
        assert mode.lower() in ["txt", "url"], "Invalid mode! ('txt' or 'url' allowed)"
        if mode.lower() == "txt":
            with open("../vocab/spanish1.txt", "r", encoding="utf-8") as f:
                self.spanish1 = set()
                for line in f:
                    self.spanish1.add(line.strip())
            with open("../vocab/spanish2.txt", "r", encoding="utf-8") as f:
                self.spanish2 = set()
                for line in f:
                    self.spanish2.add(line.strip())
        else:
            self.spanish1 = self.get_span1_vocab()
            self.spanish2 = self.get_span2_vocab()
            self.write_vocab()

        self.vocab = self.spanish1 | self.spanish2

    def get_wiki_vocab(self, url):
        """
        Given a Wikiversity Spanish URL get a set of vocabulary words/phrases

        url: (str) a URL containing list of vocabulary

        return: (set) a set of Spanish words/phrases
        """
        soup = BeautifulSoup(urlopen(url), "html.parser")
        regex = re.compile(r"<b>(.*)</b>")  # extract Spanish words

        vocab_list = [
            regex.search(str(match.find("b")))
            .group(1)
            .lower()
            .split("=")[0]
            .split("-")[0]
            .strip()
            .strip(".,-=")
            .strip()
            for match in soup.findAll("li")  # find lists
            if not len(match.attrs) and match.find("b")
        ]

        # Expand and add most of the parenthetical constructions from the scraped vocabulary
        regex = re.compile(r"\((\w+)\)")
        for elem in vocab_list:
            search = regex.search(elem)

            if search:
                start = search.start(1)
                end = search.end(1)
                match = search.group(1)

                # Pre-strip (base case)
                elem_strip = elem[: start - 1] + elem[end + 1 :]

                # If the word in parenthesis occurs at the end of the string preceded by a space,
                # it is likely intended as extra info (eg., "baile (bailar)") and can be skipped
                if elem[start - 2] == " " and len(elem) == end + 1:
                    continue

                # For constructions like "mi(s)" -> "mi" & "mis"; "tiene(n)" -> "tiene" & "tienen"
                elif match == "s" or match == "n" or match == "es":
                    sub = elem[: start - 1] + match + elem[end + 1 :]

                # For constructions like "professor(ora)" -> "professor" & "professora"
                elif match == "ora":
                    sub = elem[: start - len(match)] + match + elem[end + 1 :]

                # If the parentheses occur at the start of the string:
                # eg., "(los) estados unidos" -> "estados unidos" & "los estados unidos"
                elif start == 1:
                    sub = match + elem[end + 1 :]
                    elem_strip = elem[end + 2 :]

                # For constructions like "fantastico(a)" -> "fantastico" & "fantastica"
                elif match == "as" or match == "a" or match == "os" or match == "o":
                    sub = elem[: start - 1 - len(match)] + match + elem[end + 1 :]

                # Else case: if parentheses occur somewhere in the middle of the string:
                # eg., "té (frío) helado" -> "té helado" & "té frío helado"
                else:
                    sub = elem[: start - 1] + match + elem[end + 1 :]
                    elem_strip = elem[: start - 2] + elem[end + 1 :]

                vocab_list.append(elem_strip)
                vocab_list.append(sub)

        # Add the constituent words in a multi-word phrase to the vocabulary
        for elem in vocab_list:
            if "/" in elem:
                vocab_list.extend([e.strip("/,()") for e in elem.split("/")])
            if "-" in elem:
                vocab_list.extend([e.strip("/,()") for e in elem.split("-")])
            if " " in elem:
                vocab_list.extend([e.strip("/,()") for e in elem.split(" ")])

        return {
            word.strip()
            for word in vocab_list
            if (word != "" and "(" not in word and "/" not in word and "," not in word)
        }

    def get_span1_vocab(self):
        """
        Scrape the vocabulary of Wikiversity Spanish 1

        return: (set) a set of Spanish words/phrases
        """

        url = "https://en.wikiversity.org/wiki/Spanish_1"
        soup = BeautifulSoup(urlopen(url), "html.parser")

        tag_list = soup.find("ul").findAll(
            "a", {"href": re.compile(r"/wiki/Spanish_1/.*")}
        )
        url_list = [
            url + re.search(r'(/wiki/Spanish_1)(/.*)(" )', str(tag)).group(2)
            for tag in tag_list
            if "Linguistic_characteristics" not in str(tag)
        ]

        spanish1_vocab = set()
        for url in url_list:
            spanish1_vocab |= self.get_wiki_vocab(url)
            time.sleep(0.2)

        return spanish1_vocab

    def get_span2_vocab(self):
        """
        Scrape the vocabulary of Wikiversity Spanish 2

        return: (set) a set of Spanish words/phrases
        """

        url = "https://en.wikiversity.org/wiki/Spanish_2"
        soup = BeautifulSoup(urlopen(url), "html.parser")

        tag_list = soup.find("ul").findAll(
            "a", {"href": re.compile(r"/wiki/Spanish_2/Chapter.*")}
        )
        url_list = [
            url + re.search(r'(/wiki/Spanish_2)(/Chapter.*)(" )', str(tag)).group(2)
            for tag in tag_list
        ]

        spanish2_vocab = set()
        for url in url_list:
            spanish2_vocab |= self.get_wiki_vocab(url)
            time.sleep(0.2)

        return spanish2_vocab

    def get_gutenberg_vocab(self):
        """
        get a list of vocabulary scraped from the url given

        url: (str) an url lead to a list of vocabulary

        return: (set) a set of Spanish words
        """

        url = "https://www.gutenberg.org/files/22065/22065-h/22065-h.htm#VOCABULARY"
        soup = BeautifulSoup(urlopen(url), "html.parser")

        return {
            str(tag.string)
            for tag in soup.find("ul").findAll("b")
            if str(tag.string) != str(tag.string).upper()
        }

    def write_vocab(self):
        with open("../vocab/spanish1.txt", "w", encoding="utf-8") as fout:
            output = ""
            for word in self.spanish1:
                output += word + "\n"
            fout.write(output)

        with open("../vocab/spanish2.txt", "w", encoding="utf-8") as fout:
            output = ""
            for word in self.spanish2:
                output += word + "\n"
            fout.write(output)


class text_processor:
    """
    Tools for pre-processing texts using spaCy for downstream tasks
    """

    def __init__(self, text=""):
        self.sents = []
        self.tokens = []
        self.lemmas = []
        self.tags = []
        self.morphs = []
        self.parses = []
        self.sents = [] ### Christina - added 
        self.text = text

        if text:
            self.preprocess(self.text)
            self.spacy_pipeline(self.text)

    def preprocess(self, text):
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

    def spacy_pipeline(self, text):
        """
        Run the given text through the pretrained spaCy pipeline to extract
        sentences, tokens, lemmas, POS tags, morphology, and dependency parses
        for each sentence in the text.

        text: (str) the text (story, poem, paragraph, chapter) to process

        (no return values, the processed items are saved to lists that are
        attributes of the text_processor object)
        """
        self.sents = []
        self.tokens = []
        self.lemmas = []
        self.tags = []
        self.morphs = []
        self.parses = []
        nlp = spacy.load("es_core_news_md")
        doc = nlp(text)
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
            self.sents.append(sent.text)
            self.tokens.append(sent_tokens)
            self.lemmas.append(sent_lemmas)
            self.tags.append(sent_tags)
            self.morphs.append(sent_morphs)
            self.parses.append(sent_parses)
            self.sents.append(sent.text)        # Christina - added 

