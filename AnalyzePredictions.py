import sys
import pandas as pd

def main(predictions_file):
    df = pd.read_csv(predictions_file)

    sorted_labels = dict()
    headers = list(df)
    
    for index, row in df.iterrows():
        sorted_by_breed = \
          sorted(list(zip(headers, row[1:].tolist())), key=lambda x: x[1], reverse=True)
        print([x for x,y in sorted_by_breed[:5]])
        sorted_labels[row[0]] = sorted_by_breed
          

main(sys.argv[1])