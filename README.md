# StackOverFlow-MBDProject

## About the project
This repository implement analysis on Stack Overflow dataset to answer 2 research questions: <br>

* RQ 1. What are the popular tech topics discussed in Stack Overflow, and how has it changed over time?
* RQ 2. What are the factors contributing to the quality of questions and answers in Stack Overflow?
  * 2.1 How do tags influence the possibility of questions getting answered?
  * 2.2 Does user reputation influence the quality of Stack Overflow’s posts?
  * 2.3 Which factors drive a question to be answered (and how fast)?

The final report is available [here](<<TO UPDATE LINK>>)

## Tools and dependencies

Raw data can be downloaded from [Stack Exchange Data Dump](https://archive.org/details/stackexchange)
Raw xml data can be converted to parquet format using [this script](https://github.com/ngqhung0912/StackOverFlow-MBDProject/tree/main/codes/create_parquet)

Tools:

- Apache Spark (Pyspark, Spark-xml version 2.11-0.9.0)
- Python 
- Jupyter Notebook (For analysis and visualization, key libraries sklearn, ntlk, matplotlib)


## Main instructions: 
- Python files are located in codes/ directory, and different files answer different research questions. Since submitting 
code dependencies on Spark is not straighforward, we decided to copy-paste some of the functions between all the files - 
therefore there might be duplicates. 

- Each file will have different running instructions (for submitting to spark.) Python notebook files (.ipynb) are 
designed to run either locally or on UT's jupyter server. 

- Some figures (not all) are available on the /fig directory. Other figures will be available on their research question's
Jupyter notebook.

### Specific instructions for pca_extended.py and features_engineering.py: 

- `cd codes/` 

- Create an environment using `conda env create -f stackoverflow_env.yml --name stackoverflow_env`   

- Zip the environment: `zip -r stackoverflow_env.zip stackoverflow_env`

- Then follow the instructions on each python file's heading. 


