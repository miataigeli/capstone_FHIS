# FHIS Capstone Project
**UBC Department of French, Hispanic and Italian Studies; UBC Department of Linguistics**

## Team Members:
* Christina Park
* Darya Shyroka
* Mia Li
* Shreyas Shankar

Mentor: Jungyeul Park

## Important Links 

### Documents
- [Teamwork Contract](./docs/Teamwork_contract.md)
- [Initial Project Plan](./docs/Project_Plan.md)

### Data
- [Corpus](./corpus/): Folder containing scraped texts organized in `.json` format, labelled by source
- [Lexicons](./vocab/): Folder containing A-level Spanish vocabulary and most frequent Spanish words
- [Data splits and feature matrices](./data/): Folder containing cleaned up and organized data, ready to be fed into models

### Code
- [Text scraping](./text_scraping/): Folder containing notebooks used for scraping
- Model pipelines:
  - [Rule-based pipeline](./src/rule-based_model.ipynb)
  - [Tree-based pipeline](./src/tree_models.ipynb)
  - [SVM pipeline](./src/svm_pipeline.ipynb)
  - [BERT pipeline](./src/bert_pipeline_darya.ipynb)

### Outputs
- [Saved models](./models/): Folder containing best model checkpoints
- [Model predictions](./predictions/): Folder containing predictions made by best performing models
- [Feature distribution visualization](./visuals/): Folder containing feature distribution visualizations


### Setup Instructions
- Install the package dependencies found in [`fhis_conda_env.yml`](./fhis_conda_env.yml) using Conda:
  ```
  conda env create -f fhis_conda_env.yaml python=3.8
  conda activate fhis
  ```
  If an error(s) occur during installation they will likely occur due to the pip installation section of the YAML file. The most sure-fire solution to this seems to be to manually install the dependencies contained within the YAML file.
- Download the Spanish MCR Wordnet from [here](https://github.com/pln-fing-udelar/wn-mcr-transform).
- Download the Open Multilingual Wordnet by executing the following code within an IPython console or IPython notebook cell:
  ```
  import nltk
  nltk.download('omw')
  ```
- Follow the download instructions [here](https://stanfordnlp.github.io/CoreNLP/download.html#steps-to-setup-from-the-official-release) to install the Stanford CoreNLP client.
  - After this we strongly recommend running the [`stanford_corenlp_parser.ipynb`](./src/stanford_corenlp_parser.ipynb) notebook to better understand the CoreNLP client and to verify its installation.
- You are now ready to run some notebooks!

### Run Instructions
- Features:
  - To learn to use the feature extraction pipeline for analyzing individual texts or to generate single features, please view the notebook [`feature_pipeline_usage_tutorial.ipynb`](./src/feature_pipeline_usage_tutorial.ipynb).
  - To generate all of the feature matrices for training, validation and testing, please run the notebook [`build_feature_matrices.ipynb`](./src/build_feature_matrices.ipynb).
- Models:
  - Please ensure that you have all of the JSON data files in `./data/`, and the notebooks `src/rule-based_model.ipynb`, `src/tree_models.ipynb`, `src/svm_pipeline.ipynb` and `src/bert_pipeline_darya.ipynb`. 
  - For the rule-based model, please run `src/rule-based_model.ipynb`. 
  - For tree models, please run `src/tree_models.ipynb`.
  - For SVM, please run `svm_pipeline.ipyn`. 
  - For the neural network BERT model, please run `src/bert_pipe_darya.ipynb`. We recommend to use Google Colab to run the BERT model. Ensure that you copy `*.json` files in `data` directory into `/content/drive/MyDrive/capstone/corpus` before using the BERT model. 
