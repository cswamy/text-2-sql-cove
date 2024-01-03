# text-2-sql-cove #
This repo implements a basic version of the Chain of Verification (CoVe) approach on the Yale Spider text-to-sql challenge. Spider is a large-scale complex and cross-domain semantic parsing and text-to-SQL dataset. The goal of the Spider challenge is to develop natural language interfaces to cross-domain databases. It consists of 10,181 questions and 5,693 unique complex SQL queries on 200 databases with multiple tables covering 138 different domains.  

Chain of Verification (CoVe) paper: https://arxiv.org/abs/2309.11495

Yale Spider text-to-sql challenge: https://yale-lily.github.io/spider

## Approach ##
Implementation has four stages:
1. Baseline response (with question similarity matching): This stage uses the question in english and the correponding schema to get a baseline response from the LLM. **Important** - It uses few-shot examples of english / sql pairs. These examples are dynamically chosen from the training dataset based on the most similar questions to the question for which sql is currently being predicted. Similarity is based on 'bert-base-uncased' embeddings and cosine similarity. See `model/baseline.py`
   
2. Plan verifications: This stage uses the question in english and sql from previous stage as inputs to the LLM (along with a one-shot example). The objective of this stage is to use the LLM to interrrogate the baseline response. The expected response is a set of questions that can be used to investigate whether the columns, joins, etc. used in the baseline response sql are valid. See `build_plan() in model/planner.py`
   
3. Execute verifications: This stage uses the verification questions generated in the previous stage and the schema as inputs. The LLM is expected to answer each verification question using the schema as a knowledge base. See `execute_plan() in model/planner.py`
   
4. Final response: This uses the original english question, the schema, sql from the baseline response, and the set of verification questions and answers from stages 2 and 3. The output is a final sql query. See `model/assembler.py`

## Usage ##
1. Clone the repo with `git clone https://github.com/cswamy/text-2-sql-cove.git`
2. Navigate into repo folder and run `pip install requirements.txt`
3. Run `export OPENAI_API_KEY="<your-openai-key>"` to setup OpenAI connection
4. Create predictions against the dev dataset using `python cove.py`. 🚨 This will run predictions against the full dev spider dataset (found in `data/dev.json`). To run it against a smaller sample, use options. For e.g., `python cove.py --sample_size 100 --seed 42`
5. Output folder can be found under `experiments/` with `pred_sqls.txt` for predicted sqls, `gold_sqls.txt` for gold sqls from dev dataset and `outputs.json` file which is a list of dictionaries containing outputs for each sample in dev dataset. A `metrics.json` file is also created under `experiments/` with some useful statistics (e.g. token counts) for the run
6. Run `cd evaluation` to enter folder with evaluation scripts from Spider
7. Run `python evaluation.py --gold '<PATH TO OUTPUT FOLDER>/gold_sqls.txt' --pred '<PATH TO OUTPUT FOLDER>/pred_sqls.txt' --etype  'all' --db '../data/database/' --table '../data/tables.json' > '<PATH TO OUTPUT FOLDER>/result.txt'`
8. This will create a `result.txt` file in the output folder with evaluation metrics from Spider
9. _Optional: To analyse sql results, go into the experiments folder and run `python sql_executor.py --outputfile <PATH TO outputs.json> > <PATH TO OUTPUT FOLDER>/sql_results.txt`. This will create an `sql_results.txt` file under output folder where you can analyse sql outputs_ 

## Results ##
Results to some experiments run on the full dev dataset:

Experiment | Spider execution accuracy
| :--- | :---:
GPT-3.5-Turbo without CoVE and num_matches = 0 (baseline) | 42.4 
GPT-3.5-Turbo without CoVE and num_matches = 3  | 64.9
GPT-3.5-Turbo with CoVE and num_matches = 0  | 41.7
GPT-3.5-Turbo with CoVE and num_matches = 3  | 63.2

Notes:
1. Choosing few-shot examples in stage 1 with `bert-base-uncased` embeddings and `cosine similarity` has a 2250bps improvement over baseline
2. CoVe does not seem to add much improvements (at least with current prompts and GPT-3.5-Turbo)
3. GPT-4 was tested for a small sample (n=100, seed=42) and achieved execution accuracy of 59.0

## Potential next steps ##
1. Try with a different LLM (e.g. Code LLama)
2. Test different prompts, especially for baseline and final responses (stages 1 and 4)
