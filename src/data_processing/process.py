import json
import os

file_path = "IGPT/data/bruno/processed_data.jsonl"
output = "IGPT/data/bruno/new_processed_data.jsonl"

with open(file_path, 'r') as infile, open(output, 'w') as outfile:
            for line in infile:
                if line.strip(): 
                    record = json.loads(line)
                    record['input'] = "" 
                    json.dump(record, outfile)
                    outfile.write('\n')