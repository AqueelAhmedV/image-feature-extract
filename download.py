import pandas as pd
import os

from src.utils import download_images

df = pd.read_csv('./dataset/train.csv')

# Filter the DataFrame to include only rows where entity_name is 'item_volume'
df_item_volume = df[df['entity_name'] == 'voltage']

# Create the save folder if it doesn't exist
save_folder = './voltage_images'
os.makedirs(save_folder, exist_ok=True)

# Extract image links from the filtered DataFrame
image_links = df_item_volume['image_link'].tolist()

# Download images using the download_images function
download_images(image_links, save_folder, False)