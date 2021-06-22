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


### Run Instructions
- Features:
  - To learn to use the feature extraction pipeline for use on individual texts or to generate single features, please view the notebook [`feature_pipeline_usage_tutorial.ipynb`](./src/feature_pipeline_usage_tutorial.ipynb).
  - To generate all of the feature matrices for training, validation and testing, please run the notebook [`build_feature_matrices.ipynb`](./src/build_feature_matrices.ipynb).
- Models:
  - Please ensure that you have all of the JSON data files in `./data/`, and the notebooks `src/rule-based_model.ipynb`, `src/tree_models.ipynb`, `src/svm_pipeline.ipynb` and `src/bert_pipeline_darya.ipynb`. 
  - For the rule-based model, please run `src/rule-based_model.ipynb`. 
  - For tree models, please run `src/tree_models.ipynb`.
  - For SVM, please run `svm_pipeline.ipyn`. 
  - For the neural network BERT model, please run `src/bert_pipe_darya.ipynb`. We recommend to use Google Colab to run the BERT model. Ensure that you copy `*.json` files in `data` directory into `/content/drive/MyDrive/capstone/corpus` before using the BERT model. 
