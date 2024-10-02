# 🤖SAA : SQL Auxiliary Agent
- Text to SQL
- This model is based on the Gemma 2 Model.
- Fine tuned to use MySQL
- When you enter context and natural language, which are components of a table, a MySQL Query that matches them is returned

## Data
- [gretelai/synthetic_text_to_sql Datasets](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)

## Model
- The model was created in a total of 4 versions.<br/>
  1. [gemma2-2B-FineTuning](https://huggingface.co/SEUNGYEOPOH/SQL_Generate_Model)<br/>
  2. [gemma2-2B-FineTuning-Merge](https://huggingface.co/SEUNGYEOPOH/gemma-2-2B-Text_to_SQL-mv)<br/>
  3. [gemma2-9B-FineTuning](https://huggingface.co/SEUNGYEOPOH/gemma-2-9B-Text_to_SQL-fv)<br/>
  4. [gemma2-9B-FineTuning-Merge](https://huggingface.co/SEUNGYEOPOH/gemma-2-9B-Text_to_SQL-mv)<br/>
- Merge Version improved generalization performance by merging with Gemma2 after fine tuning.
- Choose model based on your hardware specifications!

## Set-up


## Train
