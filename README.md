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
