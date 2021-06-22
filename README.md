# FHIS Capstone Project

Team:
* Christina Park
* Darya Shyroka
* Mia Li
* Shreyas Shankar

## Links 

### Documents
- [Teamwork Contract](./docs/Teamwork_contract.md)
- [Project Plan](./docs/Project_Plan.md)

### Data
- [Corpus](./corpus/): Folder containing scraped texts organized in `.json` format by source
- [Lexicons](./vocab/): Folder containing A-level Spanish vocabulary and most frequent Spanish words
- [Data splits and feature matrices](./data/): Folder containing cleaned up and organized data ready to be fed into the model

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
- [Feature distribution visualization](./visuals/): Folder containing feature distribution visualization


### Run Instructions
- Features:
  - To use the feature pipeline to generate a single feature, please run the notebook `src/feature_pipeline_usage_tutorial.ipynb`.
  - To generate the feature matrices, please run the notebook `src/build_feature_matrices.ipynb`.
- Models:
  - Please ensure that you have `data/*.json` files and `src/rule-based_model.ipynb, tree_models.ipynb, svm_pipeline.ipynb and bert_pipe_darya.ipynb files`. 
  - For the rule-based model, please run `src/rule-based_model.ipynb`. 
  - For tree models, please run `src/tree_models.ipynb`.
  - For SVM, please run `svm_pipeline.ipyn`. 
  - For the neural network BERT model, please run `src/bert_pipe_darya.ipynb`. We recommend to use Google Colab to run the BERT model. Ensure that you copy `*.json` files in `data` directory into `/content/drive/MyDrive/capstone/corpus` before using the BERT model. 
