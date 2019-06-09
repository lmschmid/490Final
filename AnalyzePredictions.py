import sys
import pandas as pd


def main(predictions_file):
    df = pd.read_csv(predictions_file)

    sorted_labels = dict()
    headers = list(df)
    
    for index, row in df.iterrows():
        confidence_vals = map(float, row[1:].tolist())
        zipped = list(zip(headers[1:], confidence_vals))
        sorted_zip = sorted(zipped, reverse=True, key=lambda x: x[1])
        sorted_labels[row[0]] = sorted_zip

        top_5_breeds = [breed for breed,conf in sorted_zip[:5]]

        print("Top 5 breeds for image '{}': {}".format(row[0], ", ".join(top_5_breeds)))

          

main(sys.argv[1])