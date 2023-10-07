"""
This script executes the gold, baseline and final sql queries on the database.
Useful to analyse the results of an experiment.
Args:
    --outputfile (str): Path to the output json file.
    --dbpath (str): Path to the database folder.
"""
import sqlite3
import json
import argparse
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--outputfile', type=str)
parser.add_argument('--dbpath', type=str, default='../data/database/')
args = parser.parse_args()

# Get outputs json file
with open(args.outputfile, 'r') as f:
    outputs = json.load(f)

# Execute queries
for idx, output in enumerate(outputs):
    # Get db_id and gold_sql and final_sql
    db_id = output['db_id']
    gold_sql = output['gold_query']
    base_sql = output['baseline_sql']
    final_sql = output['final_sql']

    # Print output details
    print(f"Output {idx+1}")
    print(f"DB id: {db_id}")
    print(f"Question: {output['question']}")
    print(f"Gold sql: {gold_sql}")
    print(f"Base sql: {base_sql}")
    print(f"Final sql: {final_sql}")

    # Create sqllite connection and get a cursor
    conn = sqlite3.connect(f'{args.dbpath}/{db_id}/{db_id}.sqlite')
    cursor = conn.cursor()

    gold_results, base_results, final_results = [], [], []
    # Execute gold_sql
    try:
        cursor.execute(gold_sql)
        gold_results = cursor.fetchall()
    except:
        print("[ERROR] in gold sql!")
        gold_results = []
    
    # Execute baseline_sql
    try:
        cursor.execute(base_sql)
        base_results = cursor.fetchall()
    except:
        print("[ERROR] in baseline sql!")
        base_results = []

    # Execute final_sql
    try:
        cursor.execute(final_sql)
        final_results = cursor.fetchall()
    except:
        print("[ERROR] Final sql failed, trying base sql instead!")
        base_sql = output['baseline_sql']
        try:
            cursor.execute(base_sql)
            final_results = cursor.fetchall()
        except:
            print("[ERROR] in all sqls!")
            continue

    # Close cursor and connection
    cursor.close()
    conn.close()

    # Print results
    print(f"\nGold results: {gold_results}\n")
    # print(f"Base results: {base_results}\n")
    print(f"Final results: {final_results}")
    print("--------------------------------------------------\n")
