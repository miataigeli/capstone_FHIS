Spanish Scraping Resources:

- Reading texts by reading level: https://lingua.com/spanish/reading/, also the pdfs my mom sent me are A1 level (Aventura Joven)
- Will likely need a PDF scraper

NLP resources: 

- Spanish BERT: https://medium.com/dair-ai/beto-spanish-bert-420e4860d2c6
	- can be used for POS tagging (and potentially finding the verb constructions mentioned), machine translation (for the "hard" words), and text classification tasks 
(such as determining text level?), information retrieval?
- There are HuggingFace libraries that make it easier to use transformer-based pre-trained models
- List of Spanish resources: https://github.com/dav009/awesome-spanish-nlp


How to evaluate Text Readability with NLP: https://medium.com/glose-team/how-to-evaluate-text-readability-with-nlp-9c04bd3f46a2

Things left to decide:
- What type of model we should use - rule-based, transformer, other?
	-> I think BETO (Spanish BERT) might work well 
