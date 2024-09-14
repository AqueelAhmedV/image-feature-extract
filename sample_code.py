import os
import random
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import models, transforms
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from src.utils import download_images, parse_string
from src.constants import entity_unit_map

def extract_features(image_path):
    try:
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            features = model(img_tensor)
        
        return features.squeeze().numpy()
    except (OSError, UnidentifiedImageError):
        print(f"Skipping corrupted image: {image_path}")
        return None  # or a default feature vector

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predictor(image_link, category_id, entity_name):
    # Load or create the model (you might want to do this outside the function for efficiency)
    DATASET_FOLDER = './dataset/'
    IMAGE_FOLDER = './images/'
    
    train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    
    # Extract features for training data
    train_features = [f for f in [extract_features(os.path.join(IMAGE_FOLDER, os.path.basename(link))) for link in train_df['image_link']] if f is not None]
    
    # Prepare labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['entity_value'])
    
    # Train model
    model = train_model(train_features, train_labels)
    
    # Extract features for the current image
    image_path = os.path.join(IMAGE_FOLDER, os.path.basename(image_link))
    features = extract_features(image_path)
    
    # Make prediction
    prediction = model.predict([features])[0]
    
    # Decode prediction and format output
    decoded_prediction = label_encoder.inverse_transform([prediction])[0]
    
    # Get appropriate unit for the entity
    unit = random.choice(list(entity_unit_map[entity_name]))
    
    return f"{decoded_prediction} {unit}"

if __name__ == "__main__":
    DATASET_FOLDER = './dataset/'
    IMAGE_FOLDER = './images/'

    
    
    # Ensure images are downloaded
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    # download_images(test['image_link'], IMAGE_FOLDER)
    

    # train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    # download_images(train['image_link'], IMAGE_FOLDER)

    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out_1.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)