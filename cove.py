import random
import json
import os
import argparse
import math
import pandas as pd
from pathlib import Path
from scripts import utils
from model import baseline, planner, assembler
from timeit import default_timer as timer
import subprocess
from transformers import AutoTokenizer, AutoModelForTextEncoding
import torch

def predict_sql(train_data, train_embeddings, test_embedding, schemas, db_id, question, num_matches, model):
    """
    Predict SQL for a given question and database ID
    """
    # Set OpenAI model version
    os.environ['MODEL'] = model

    # Get top_n similar questions from training dataset
    num_matches = num_matches
    top_matches = utils.get_similar_questions(train_data, train_embeddings, test_embedding, num_matches)

    # Variable to track tokens across LLM calls
    input_tokens = 0
    output_tokens = 0
    start_time = timer()

    # Get baseline response from model
    baseline_response = baseline.get_baseline_response(question, db_id, top_matches, schemas)
    if baseline_response is None:
        return None
    input_tokens += baseline_response['input_tokens']
    output_tokens += baseline_response['output_tokens']
    baseline_sql = baseline_response['sql']
    target_schema = baseline_response['target_schema']

    # Planner
    planner_response = planner.build_plan(question, baseline_sql)
    if planner_response is None:
        return None
    input_tokens += planner_response['input_tokens']
    output_tokens += planner_response['output_tokens']
    plan = planner_response['plan']

    # Execute plan
    exec_response = planner.execute_plan(target_schema, plan)
    if exec_response is None:   
        return None
    input_tokens += exec_response['input_tokens']
    output_tokens += exec_response['output_tokens']
    qa_str = exec_response['qa']

    # Assemble final response
    final_response = assembler.assemble_output(question, target_schema, baseline_sql, qa_str)
    if final_response is None:
        return None
    input_tokens += final_response['input_tokens']
    output_tokens += final_response['output_tokens']
    final_sql = final_response['final_sql']
    end_time = timer()

    # Return object
    return {
        'baseline_sql': baseline_sql,
        'final_sql': final_sql,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'time': end_time - start_time
    }

if __name__ == "__main__":

    # Prevent OS from sleeping during execution
    caffeinate_process = subprocess.Popen(['caffeinate', "-i"])

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--dbpath", type=str, default="data/database/")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    SEED = args.seed
    SAMPLE_SIZE = args.sample_size
    DB_PATH = args.dbpath

    # Add elements to list below for experiments
    NUM_MATCHES_LIST = [3]
    MODEL_LIST = ['gpt-3.5-turbo']

    # Get training dataset
    # Filter to only keep db_id, question, question_toks, and query
    full_train_data = utils.load_dataset(DATA_DIR, dataset='train')
    keys_to_keep = ['db_id', 'question', 'question_toks', 'query']
    train_data = [{k: v for k, v in d.items() if k in keys_to_keep} for d in full_train_data]

    # Load schemas from Spider table.json
    schemas = utils.load_schemas(DATA_DIR)

    # Get tokenizer and model
    checkpoint = 'bert-base-uncased'
    max_seq_length = 50
    emb_tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    emb_model = AutoModelForTextEncoding.from_pretrained(checkpoint)

    # Set hf tokenizer parallelism to false
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # Load embeddings for training questions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_embeddings = torch.load(
        'model/training_ques_embeddings.pt',
        map_location=torch.device(device))

    # Get dev dataset
    dev_data = utils.load_dataset(DATA_DIR, dataset='dev')
    if SEED is not None:
        random.seed(SEED)
    if SAMPLE_SIZE is None:
        samples = dev_data
    else:
        samples = random.sample(dev_data, SAMPLE_SIZE)

    # Predict SQL 
    # Object returned contains baseline_sql, final_sql, input_tokens, output_tokens, 
    # and time
    for MODEL in MODEL_LIST:
        for NUM_MATCHES in NUM_MATCHES_LIST:
            print(f"[INFO] Predicting SQLs with top_n={NUM_MATCHES} and model={MODEL}")
            outputs = []
            for idx, sample in enumerate(samples):
                print(f"[INFO] Predicting SQL for question {idx+1} of {len(samples)}")
                db_id = sample['db_id']
                question = sample['question']
                # Tokenize question
                input = emb_tokenizer(
                    question, 
                    padding='max_length',
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors='pt')
                # Get embedding for question
                emb_model.eval()
                with torch.inference_mode():
                    emb_output = emb_model(**input)
                    test_embedding = emb_output.last_hidden_state[:,0,:]
                # Predict sql
                output = predict_sql(
                    train_data, 
                    train_embeddings,
                    test_embedding,
                    schemas, 
                    db_id, 
                    question, 
                    NUM_MATCHES, 
                    MODEL)
                if output is None:
                    break
                output['question'] = question
                output['db_id'] = db_id
                output['gold_query'] = sample['query']
                outputs.append(output)

            # Write gold_queries with db_id to file
            output_dir = Path("experiments/" + "topn_" + str(NUM_MATCHES) + "_model_" + MODEL + "_seed_" + str(SEED) + "_sample_" + str(SAMPLE_SIZE))
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(output_dir) + "/gold_sqls.txt"
            with open(filepath, "w") as f:
                for output in outputs:
                    f.write(output['gold_query'] + "\t" + output['db_id'] + "\n")
                print(f"[INFO] Gold sqls written to txt file!")

            # Write predicted sql to file
            filepath = str(output_dir) + "/pred_sqls.txt"
            with open(filepath, "w") as f:
                for output in outputs:
                    # Uncomment below 4 lines to get baseline performance without CoVe planning 
                    # baseline_str = output['baseline_sql']
                    # start_idx = baseline_str.lower().find('select')
                    # baseline_sql = baseline_str[start_idx:].replace('\n', ' ').replace('\t', ' ').replace('  ', ' ').strip()
                    # f.write(baseline_sql + "\n")
                    
                    # Uncomment below line to get CoVe performance. Make sure to comment out the 4 lines above
                    f.write(output['final_sql'] + "\n")
                print(f"[INFO] Predicted sqls written to txt file!")

            # Write outputs to file
            filepath = str(output_dir) + "/outputs.json"
            with open(filepath, "w") as f:
                json.dump(outputs, f, indent=4)
                print(f"[INFO] Output file written to json!")
        
            # Create dataframe
            df = pd.DataFrame(outputs)
            
            metrics = {}
            metrics['avg_input_tokens'] = math.ceil(df['input_tokens'].mean())
            metrics['avg_output_tokens'] = math.ceil(df['output_tokens'].mean())
            metrics['avg_time'] = round(df['time'].mean(), 2)
            metrics['sample_size'] = SAMPLE_SIZE
            metrics['seed'] = SEED
            metrics['topn'] = NUM_MATCHES
            metrics['model'] = MODEL
            
            # Write metrics to file
            filepath = "experiments/metrics.json"
            if Path(filepath).is_file():
                with open(filepath) as f:
                    data = json.load(f)
                data.append(metrics)
            else:
                data = [metrics]
            with open(filepath, "w") as f:
                json.dump(data, f, indent=4)   
                print(f"[INFO] Metrics written to {filepath}\n")

    # Kill caffeinate process
    if caffeinate_process.poll() is None:
        caffeinate_process.terminate()
