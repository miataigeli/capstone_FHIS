import os
import re
import json
import time
from collections import defaultdict
from bs4 import BeautifulSoup
from urllib.request import urlopen


def read_corpus(path):
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
                    corpus[d["level"]].append(d)
    return corpus


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
            url
            + re.search(r'(/wiki/Spanish_2)(/Chapter.*)(" )', str(tag)).group(2)
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
