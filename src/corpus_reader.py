import os
import json
from collections import defaultdict


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

    corpus: (dict{list[dicts]}) a dictionary of texts arranged by reading level
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
