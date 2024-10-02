# 🤖SAA : SQL Auxiliary Agent
- gemma spirnt
- Text to SQL
- This model is based on the Gemma 2 Model.
- Fine tuned to use MySQL
- When you input context and natural language, a matching MySQL query is generated.

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white"> <img src="https://img.shields.io/badge/Hugging Face-FFD21E?style=for-the-badge&logo=Hugging Face&logoColor=black"> <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white"> <img src="https://img.shields.io/badge/Gemma2-8E75B2?style=for-the-badge&logo=Google Gemini&logoColor=white"> <img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=MySQL&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">

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
### 1. Guide
- In the absence of a guide, queries or DDL from other engines are often used. To solve this problem, I created a guide and trained it by adding information from the template. If you want more elaborate results, it would be helpful if you modified the guide specifically or provided examples.
```python
guide = '''
    You are a MySQL expert. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.
    
    You must output the SQL query that answers the question.
    
    Let me give you some precautions:

    1. Return ONLY the generated SQL query and nothing else.

    2. Do not use or add any information outside of the context provided.

    3. Pay close attention to which column belongs to which table. If the context contains more than one table, create a query by performing a JOIN operation using the common column

    4. Ensure that you do not query for columns that do not exist in the tables, and use aliases only where required.

    5. When asked about averages (using the AVG function) or ratios, make sure to use the appropriate aggregation function.

    6. Pay close attention to the filtering criteria mentioned in the question and incorporate them using the WHERE clause in your SQL query.

    7. If the question involves multiple conditions, use logical operators such as AND or OR to combine them effectively.

    8. When dealing with date or timestamp columns, use appropriate date functions (e.g., DATE_DIFF, DATE_ADD) for extracting specific parts of the date or performing date arithmetic.

    9. If the question involves grouping of data (e.g., finding totals or averages for different categories), use the GROUP BY clause along with appropriate aggregate functions.

    10. Consider using aliases for tables and columns to improve the readability of the query, especially in the case of complex joins or subqueries.
    
    Additional guidelines :
    
    - Use LIMIT to restrict the number of returned rows when the question asks for a limited number of results or pagination.

    - Use DISTINCT to remove duplicates if the question requests unique values.

    - Use ORDER BY to sort the results in ascending (ASC) or descending (DESC) order, as specified in the question.

    - Handle NULL values appropriately. Use IS NULL or IS NOT NULL to filter out or include NULL values, and use the COALESCE function when needed to replace NULLs.

    - Use HAVING instead of WHERE when filtering aggregated data.

    - Use the CASE statement for conditional logic to return different values based on specific conditions.

    - Leverage subqueries when the question is complex and requires filtering or extracting data from nested queries.

    - Use database-specific functions like STR_TO_DATE() in MySQL or TO_CHAR() in PostgreSQL when appropriate.

    - Optimize JOIN performance by ensuring unnecessary data is filtered in the WHERE clause and considering the order of JOINs for efficiency.

    - Use ROUND() or similar functions for numeric accuracy when handling decimals or rounding values.

    - Interpret vague questions to produce logical queries, such as converting "top few items" into a specific number like "top 5" based on context.

'''
```
### 2. Temp Generate
- query : query refers to a natural language.
  - Ex. What is the total number of employees in the Employee table?
  
- Context : Context refers to structural information such as table name and columns. / I have assigned a schema query.
  - Ex. Create table If Not Exists Products (product_id int, low_fats ENUM('Y', 'N'), recyclable ENUM('Y','N')) insert into Products (product_id, low_fats, recyclable) values ('0', 'Y', 'N')  
```python
def eval_com(query : str, context : str, guide : str, model, tokenizer, max_tokens=1000) -> str:
    device = "cuda:0"


    prompt_tem = '''
    <bos><start_of_turn>user
    {guide}
    
    context: {context}

    question: {query}

    <end_of_turn>
    <start_of_turn>model
    '''


    prompt = prompt_tem.format(query=query, context = context, guide = guide)

    encoders = tokenizer(prompt, return_tensors="pt",add_special_tokens=True)

    model_inputs = encoders.to(device)

    generated_ids = model.generate(
      **model_inputs,
      max_new_tokens=max_tokens,
      do_sample=True)
      # pad_token_id=tokenizer.eos.token_id)

    decode = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return (decode)
```
### 3. BitsAndBytes
- Loads the model weights in 4-bit precision to reduce memory usage and optimize performance.
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16

)
```
### 4. LoRA (Low-Rank Adaptation)
- By using BitsAndBytes to select only the Linear layers that have been quantized to 4-bit, you can apply fine-tuning specifically to those layers. This approach reduces memory usage while efficiently training only the necessary parts of the model, without the need to fine-tune the entire model.
```python
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

## Train Report
- [My report](https://wandb.ai/dhwmd08-tech-university-of-korea/huggingface/reports/Gemma2-9B-Train-Report--Vmlldzo5NTM5NTEz?accessToken=uwuvw9ugbz4ggglpt4zre8meecuz5vp425meh77ciqzmbhysmampso9jzstm7msf)


Let's Try~!

