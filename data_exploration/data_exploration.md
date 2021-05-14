**Spanish vocab websites**:

- Wiki pages on 1000 most common Spanish words: https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Spanish1000
- top 2000 Spanish words: https://app.memrise.com/course/132342/top-2000-words-in-spanish/ 
- 5000 words categorized thematically:  https://app.memrise.com/course/203799/5000-most-frequent-spanish-words/
- 5000 Spanish words (Flash cards): https://www.brainscape.com/packs/top-5000-spanish-words-8804899

**PDF reader**:
    
   ```
   
   from tika import parser # pip install tika

    raw = parser.from_file('sample.pdf')
    print(raw['content'])

```

Need to have Java runtime installed.
There are other resources available but they do not have good reputation amongst users: https://towardsdatascience.com/pdf-preprocessing-with-python-19829752af9f

**Spacy Support for Spanish**: 

- lemmatizer, sentence segmentation (sentence recognizer), parser, NER, morphologizer available.
- based on newswire texts