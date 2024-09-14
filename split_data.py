import pandas as pd
import os

def split_csv(input_file, ocr_output, other_output):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Define the entity names for 'other' category
    other_entities = ['height', 'width', 'depth']
    
    # Split the dataframe
    df_other = df[df['entity_name'].isin(other_entities)]
    df_ocr = df[~df['entity_name'].isin(other_entities)]
    
    # Save the split dataframes
    df_ocr.to_csv(ocr_output, index=False)
    df_other.to_csv(other_output, index=False)
    
    print(f"Split {input_file}:")
    print(f"  OCR rows: {len(df_ocr)}")
    print(f"  Other rows: {len(df_other)}")

# Define input and output file paths
input_train = './dataset/train.csv'
input_test = './dataset/test.csv'

output_ocr_train = './dataset/ocr_train.csv'
output_other_train = './dataset/other_train.csv'
output_ocr_test = './dataset/ocr_test.csv'
output_other_test = './dataset/other_test.csv'


# Split train.csv
split_csv(input_train, output_ocr_train, output_other_train)

# Split test.csv
split_csv(input_test, output_ocr_test, output_other_test)

print("Splitting complete.")
