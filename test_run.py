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
import joblib

def extract_features(image_path):
    try:
        # Load the model only once and move it to GPU if available
        if not hasattr(extract_features, 'model'):
            extract_features.model = models.resnet50(pretrained=True)
            extract_features.model = torch.nn.Sequential(*list(extract_features.model.children())[:-1])
            extract_features.device = torch.device("cuda:0")
            extract_features.model.to(extract_features.device)
            extract_features.model.eval()

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        img_tensor = img_tensor.to(extract_features.device)
        
        with torch.no_grad():
            features = extract_features.model(img_tensor)
        
        return features.squeeze().cpu().numpy()
    except (OSError, UnidentifiedImageError):
        print(f"Skipping corrupted image: {image_path}")
        return None
    except RuntimeError as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predictor(image_link, category_id, entity_name):
    DATASET_FOLDER = './dataset/'
    TRAIN_IMAGE_FOLDER = './train_images/'
    TEST_IMAGE_FOLDER = './test_images/'
    MODEL_PATH = './model/random_forest_model.joblib'
    LABEL_ENCODER_PATH = './model/label_encoder.joblib'
    
    # Load or train the model
    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_ENCODER_PATH):
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    else:
        train_df = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
        
        # Extract features for training data
        train_features = [f for f in [extract_features(os.path.join(TRAIN_IMAGE_FOLDER, os.path.basename(link))) for link in train_df['image_link']] if f is not None]
        
        # Prepare labels
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(train_df['entity_value'])
        
        # Train model
        model = train_model(train_features, train_labels)
        
        # Save model and label encoder
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    # Extract features for the current image
    image_path = os.path.join(TEST_IMAGE_FOLDER, os.path.basename(image_link))
    features = extract_features(image_path)
    
    if features is None:
        return "Unable to process image"
    
    # Make prediction
    prediction = model.predict([features])[0]
    
    # Decode prediction and format output
    decoded_prediction = label_encoder.inverse_transform([prediction])[0]
    
    # Get appropriate unit for the entity
    unit = random.choice(list(entity_unit_map[entity_name]))
    
    return f"{decoded_prediction} {unit}"

if __name__ == "__main__":
    DATASET_FOLDER = './dataset/'
    TRAIN_IMAGE_FOLDER = './train_images/'
    TEST_IMAGE_FOLDER = './test_images/'

    # Load and sample train data
    train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    train_sample = train.sample(n=1000, random_state=42)
    
    # Load and sample test data
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    test_sample = test.sample(n=500, random_state=42)
    
    # Ensure images are downloaded (uncomment if needed)
    # download_images(train_sample['image_link'], TRAIN_IMAGE_FOLDER)
    # download_images(test_sample['image_link'], TEST_IMAGE_FOLDER)

    # Set the torch.cuda.empty_cache() before processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_sample['prediction'] = test_sample.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out_1.csv')
    test_sample[['index', 'prediction']].to_csv(output_filename, index=False)