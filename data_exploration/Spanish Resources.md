Spanish Graded Reader Resources

Scraping: 

- Reading texts by reading level: https://lingua.com/spanish/reading/, also Aventura Joven is A1 level (I have 3 pdfs)
- Will likely need a PDF scraper. I tried some out here: https://colab.research.google.com/drive/1jms2k8tB2ORgZGUFs7j5rEumn141a5JU?usp=sharing


NLP resources:

- Spanish BERT: https://medium.com/dair-ai/beto-spanish-bert-420e4860d2c6
    - can be used for POS tagging (and potentially finding the verb constructions mentioned), machine translation (for the "hard" words), and text classification tasks
(such as determining text level?), information retrieval?
- There are HuggingFace libraries that make it easier to use transformer-based pre-trained models
- List of Spanish resources: https://github.com/dav009/awesome-spanish-nlp
- TextStat, can provide Flesch Reading Ease Score for Spanish (which we could use as one of our features): https://pypi.org/project/textstat/ 


NLP for Evaluating Text Readability, Literature Review: 

How to evaluate Text Readability with NLP:
https://medium.com/glose-team/how-to-evaluate-text-readability-with-nlp-9c04bd3f46a2 

Automatic Text Difficulty Classifier (paper in which NLP methods were used to classify Portuguese texts as A1, A2, B1, B2, and C1): 

Link to pdf: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwisi8CZpLPwAhUKWK0KHXB4Ad0QFjAPegQICRAD&url=http%3A%2F%2Fwww.inesc-id.pt%2Fpublications%2F11043%2Fpdf&usg=AOvVaw1YwUH-2Yixi14ScOVTzoIX

Some main points:
 they used a lot of gold standard annotated data which we would need to scrape if we were to use this method
the extraction of linguistic features from texts is a core task to classify by reading level
need to take lexical difficulty (vocabulary level) and syntactic difficulty (sentence complexity, verb tenses?) into account
 so just operating based on frequency lists won't work
STRING used to extract features - performs allthe basic NLP tasks, namely tokenization and text segmentation, part-of-speech tagging, rule-based and statistical morphosyntactic disambiguation, shallow parsing (chunking) and deep parsing (dependency extraction)
YAH (Yet Another Hyphenator) used to extract the number of syllables
both STRING and YAH are Portuguese-specific, so we’d need to find or create equivalents for Spanish
52 features extracted by the system, 7 categories (found in Appendix):
POS tags: ADJ, ADV, ART, CONJ, INTERJ, NOUN, NUM, PASTPART, PREP, PRON, PUNCT, SYMBOL
Chunks: NP, AP, PP, ADVP, VTEMP, VASP, VMOD, VCOP, VPASTPART, VGER, VINF, VF, SC
Word and sentence features: number of sentences, number of words, number of unique words, word frequencies
Verb features: number of unique verb forms, number of auxiliary verbs, number of main verbs
Different metrics involving averages and frequencies
Metrics involving syllables
Extra features: total number of dependencies, total number of tree nodes, etc
Note: Dependency parsing!
The corpus used to train the classifier consists of 237 texts, exams and materials used for teaching European Portuguese as a foreign language -> might be a good idea for us to use Spanish textbooks
ML algorithms available in WEKA were tested, the best-performing learning algorithm was LogitBoost, with a root mean square error of 0.269. Note: I don’t think any neural methods were tested. WEKA: http://old-www.cms.waikato.ac.nz/ml/weka/

Linguistic Features for Readability Assessment (paper): 
https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwisi8CZpLPwAhUKWK0KHXB4Ad0QFjASegQIChAD&url=https%3A%2F%2Fwww.aclweb.org%2Fanthology%2F2020.bea-1.1.pdf&usg=AOvVaw2KAJBmAS_S2sxKapnYIKmp
Incorporates Linguistics Features with Neural Models
Used the single numerical output of a neural model as one of the features, and joined it with the linguistic features, and fed these features as input into one of the simpler non-neural models. 
Word embeddings of the text as input to the neural model
All experiments involved 5-fold cross validation. 
Models tested: SVMs, Linear Models and Logistic Regression, CNN (Convolutional Neural Networks), Transformers (pretrained BERT model sourced from the HuggingFace transformers library), HAN (Hierarchical attention network): two bidirectional RNNs each with a separate attention mechanism - one that attends to different words within each sentence, and one that attends to the sentences within the document -> thought to better mimic the structure of documents. 
Conclusion: addition of linguistic features does not improve state-of-the-art deep learning models. However, in low-resource settings, they can improve the performance of deep learning models. Similarly, with more diverse and more accurately and consistently labeled corpora, the linguistic features could prove more useful.


Computational Assessment of Text Readability:
A Survey of Current and Future Research (2014): https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjEmYHnr7PwAhU7FzQIHTEPBIE4ChAWMAJ6BAgGEAM&url=http%3A%2F%2Fwww-personal.umich.edu%2F~kevynct%2Fpubs%2FITL-readability-invited-article-v10-camera.pdf&usg=AOvVaw0pq1ij9Mk5JX8ZS9RlX52k

Text Complexity (Natural Language) (2018): https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjEmYHnr7PwAhU7FzQIHTEPBIE4ChAWMAN6BAgCEAM&url=http%3A%2F%2Fcs229.stanford.edu%2Fproj2018%2Freport%2F185.pdf&usg=AOvVaw1YKCr1ccb8_89V04eZjOf4 
Tested different models with linguistic features, LSTM did best -> Transformers would probably do better but this was 2018


Main takeaways from Literature Review:
Neural methods such as Transformers and RNNs perform better than linguistic-feature based ML algorithms.


Things left to decide:
- What type of model we should use - rule-based, transformer, other?
    -> I think BETO (Spanish BERT) might work well
-    Note: “BERT is pretrained as a language
model, [therefore] it tends to rely more on semantic
than structural differences during the classification
phase and therefore performs better on problems
with distinct semantic differences between readability
classes”

