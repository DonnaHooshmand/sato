import csv
import json

csv_file = 'edstays.csv'
jsonl_file = 'edstays.jsonl'

with open(csv_file, newline='') as f:
    reader = csv.reader(f)
    rows = list(reader)

header = rows[0]
data_rows = rows[1:]

with open(jsonl_file, 'w') as out:
    json.dump({
        'id': '0',
        'header': header,
        'rows': data_rows
    }, out)
    out.write('\n')
