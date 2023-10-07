import os
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForTextEncoding
from torch.nn.functional import cosine_similarity
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

def get_model():
    """
    Function to set LLM parameters
    """
    return {
        'model_name': os.environ['MODEL'],
        'temperature': 0,
        'max_tokens': 1000,
        'top_p': 1,
        'frequency_penalty': 0,
        'presence_penalty': 0
    }

def load_schemas(data_dir:str):
    """
    Function to load schemas from Spider dataset
    """
    schema_path = os.path.join(data_dir, 'tables.json')
    try:
        with open(schema_path) as f:
            schemas = json.load(f)
        return schemas
    except:
        print(f"[ERROR] Failed to load schemas from {schema_path}.")
        return None

def load_dataset(data_dir:str, dataset:str='test'):
    """
    Function to load training dataset from Spider dataset
    """
    if dataset == 'train':
        path = os.path.join(data_dir, 'train_spider.json')
    elif dataset == 'dev':
        path = os.path.join(data_dir, 'dev.json')
    elif dataset == 'test':
        path = os.path.join(data_dir, 'test.json')
    else:
        print(f"[ERROR] Invalid dataset {dataset}. Valid options are 'train', 'dev', and 'test'.")
    try:
        with open(path) as f:
            dataset = json.load(f)
        return dataset
    except:
        print(f"[ERROR] Failed to load data from {path}.")
        return None

def generate_schema_code(db_id:str, schemas:dict):
    """
    Given a database ID and a dictionary of schemas, 
    this function builds and returns the schema code for the specified database.
    
    Args:
    - db_id (str): The ID of the database for which to build the schema code.
    - schemas (dict): A dictionary containing the schema information for all databases.
    
    Returns:
    - schema_code (str): The schema code for the specified database.
    """
    # Get schema for db_id
    schema = [schema for schema in schemas if schema['db_id'] == db_id][0]
    # Initialize schema code
    schema_code = ""
    # Iterate over tables
    for table_idx, table_name in enumerate(schema['table_names_original']):
        schema_code += f"CREATE TABLE \"{table_name}\" (\n"
        primary_keys = []
        foreign_keys = []
        # Iterate over columns
        for column_idx, column_name in enumerate(schema['column_names_original']):
            if column_name[0] == table_idx:
                schema_code += f"\"{column_name[1]}\" {schema['column_types'][column_idx]},\n"
                # Check if column is a primary key
                if column_idx in schema['primary_keys']:
                    primary_keys.append(column_name[1])
                # Check if column is a foreign key
                for fk in schema['foreign_keys']:
                    if column_idx == fk[0]:
                        foreign_keys.append((column_name[1], schema['column_names_original'][fk[1]][0], schema['column_names_original'][fk[1]][1]))
        # Add primary keys
        if len(primary_keys) > 0:
            schema_code += f"PRIMARY KEY (\"{', '.join(primary_keys)}\"),\n"
        # Add foreign keys
        if len(foreign_keys) > 0:
            for fk in foreign_keys:
                schema_code += f"FOREIGN KEY (\"{fk[0]}\") REFERENCES \"{schema['table_names_original'][fk[1]]}\"(\"{fk[2]}\"),\n"
        # Remove trailing comma and close table with a bracket
        schema_code = schema_code[:-2] + "\n);\n\n"
    return schema_code

def get_similar_questions(train_data, train_embeddings, test_embedding, top_n):
    """
    Function to get the top n most similar questions from the training dataset.
    """
    # Compute cosine similarity between test question and training questions
    similarities = [cosine_similarity(test_embedding, emb.unsqueeze(0)) for emb in train_embeddings]

    # Get top n most similar questions
    top_matches = []
    top_indices = torch.argsort(torch.cat(similarities, dim=0), descending=True)[:top_n].tolist()
    for idx in top_indices:
        top_match = {}
        top_match['idx'] = idx
        top_match['question'] = train_data[idx]['question']
        top_match['query'] = train_data[idx]['query']
        top_match['db_id'] = train_data[idx]['db_id']
        top_matches.append(top_match)
    
    return top_matches

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai(
    model:str, 
    messages:list, 
    temperature:int,
    max_tokens:int,
    top_p:int,
    frequency_penalty:int,
    presence_penalty:int):
    """
    Function to make API call to OpenAI API with exponential backoff
    """
    return openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
        )