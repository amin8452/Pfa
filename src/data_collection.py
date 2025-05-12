import os
import argparse
import json
import shutil
import requests
from PIL import Image, ImageFilter
from tqdm import tqdm
import numpy as np
import random
import glob
import importlib
import io
import zipfile
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Data collection and preparation for glasses 3D reconstruction')
    parser.add_argument('--data_root', type=str, default='data/glasses', help='Root directory for data')
    parser.add_argument('--scrape_images', action='store_true', help='Scrape glasses images from the web')
    parser.add_argument('--process_images', action='store_true', help='Process and normalize collected images')
    parser.add_argument('--augment_data', action='store_true', help='Augment existing data')
    parser.add_argument('--split_data', action='store_true', help='Split data into train/val/test')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation data')
    parser.add_argument('--num_images', type=int, default=1000, help='Number of images to scrape')
    parser.add_argument('--image_sources', type=str, nargs='+',
                        default=['eyeglasses', 'sunglasses', 'reading_glasses', 'sports_glasses',
                                'designer_glasses', 'vintage_glasses', 'rimless_glasses', 'kids_glasses'],
                        help='Types of glasses to scrape')
    parser.add_argument('--download_dataset', type=str, default=None,
                        help='Download a pre-existing dataset (options: kaggle, google, custom_url)')
    parser.add_argument('--dataset_url', type=str, default=None,
                        help='URL for custom dataset download')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for processing large datasets')
    return parser.parse_args()

def scrape_images(data_root, num_images, image_sources):
    """
    Scrape glasses images from various sources using Unsplash API
    """
    raw_dir = os.path.join(data_root, 'raw_images')
    os.makedirs(raw_dir, exist_ok=True)

    # Search terms for different types of glasses
    search_terms = {
        'eyeglasses': ['eyeglasses', 'prescription glasses', 'optical frames', 'glasses frames'],
        'sunglasses': ['sunglasses', 'shades', 'sun glasses', 'aviator sunglasses', 'wayfarer sunglasses'],
        'reading_glasses': ['reading glasses', 'computer glasses', 'blue light glasses'],
        'sports_glasses': ['sports glasses', 'athletic eyewear', 'sports goggles', 'ski goggles'],
        'designer_glasses': ['designer glasses', 'luxury eyewear', 'fashion glasses', 'branded glasses'],
        'vintage_glasses': ['vintage glasses', 'retro glasses', 'classic eyewear', 'old-fashioned glasses'],
        'rimless_glasses': ['rimless glasses', 'frameless glasses', 'minimal glasses'],
        'kids_glasses': ['kids glasses', 'children eyewear', 'youth glasses', 'pediatric glasses']
    }

    # Create a metadata file to track the sources
    metadata = []

    # Use a free image API (Pexels)
    pexels_api_key = "YOUR_PEXELS_API_KEY"  # Replace with your API key or leave empty to use fallback

    # Fallback URLs for each category (pre-selected free images)
    fallback_urls = {
        'eyeglasses': [
            "https://images.pexels.com/photos/701877/pexels-photo-701877.jpeg",
            "https://images.pexels.com/photos/947885/pexels-photo-947885.jpeg",
            "https://images.pexels.com/photos/1054777/pexels-photo-1054777.jpeg",
            "https://images.unsplash.com/photo-1574258495973-f010dfbb5371",
            "https://images.unsplash.com/photo-1577803645773-f96470509666"
        ],
        'sunglasses': [
            "https://images.pexels.com/photos/46710/pexels-photo-46710.jpeg",
            "https://images.pexels.com/photos/701877/pexels-photo-701877.jpeg",
            "https://images.pexels.com/photos/1362558/pexels-photo-1362558.jpeg",
            "https://images.unsplash.com/photo-1511499767150-a48a237f0083",
            "https://images.unsplash.com/photo-1473496169904-658ba7c44d8a"
        ],
        'reading_glasses': [
            "https://images.pexels.com/photos/1438081/pexels-photo-1438081.jpeg",
            "https://images.pexels.com/photos/907862/pexels-photo-907862.jpeg",
            "https://images.pexels.com/photos/2095953/pexels-photo-2095953.jpeg",
            "https://images.unsplash.com/photo-1591076482161-42ce6da69f67",
            "https://images.unsplash.com/photo-1574258495973-f010dfbb5371"
        ],
        'sports_glasses': [
            "https://images.pexels.com/photos/33622/fashion-model-beach-hat.jpg",
            "https://images.pexels.com/photos/1342609/pexels-photo-1342609.jpeg",
            "https://images.pexels.com/photos/1687719/pexels-photo-1687719.jpeg",
            "https://images.unsplash.com/photo-1609902726285-00668009f004",
            "https://images.unsplash.com/photo-1577744062836-dde739cb8089"
        ],
        'designer_glasses': [
            "https://images.pexels.com/photos/2228559/pexels-photo-2228559.jpeg",
            "https://images.pexels.com/photos/2765894/pexels-photo-2765894.jpeg",
            "https://images.pexels.com/photos/2587464/pexels-photo-2587464.jpeg",
            "https://images.unsplash.com/photo-1583394838336-acd977736f90",
            "https://images.unsplash.com/photo-1546180572-28e937c8128b"
        ],
        'vintage_glasses': [
            "https://images.pexels.com/photos/1499480/pexels-photo-1499480.jpeg",
            "https://images.pexels.com/photos/2690323/pexels-photo-2690323.jpeg",
            "https://images.pexels.com/photos/1192609/pexels-photo-1192609.jpeg",
            "https://images.unsplash.com/photo-1577803645773-f96470509666",
            "https://images.unsplash.com/photo-1582142407894-ec8a1f6b0f6b"
        ],
        'rimless_glasses': [
            "https://images.pexels.com/photos/2589653/pexels-photo-2589653.jpeg",
            "https://images.pexels.com/photos/2589652/pexels-photo-2589652.jpeg",
            "https://images.pexels.com/photos/2599245/pexels-photo-2599245.jpeg",
            "https://images.unsplash.com/photo-1584036553516-bf83210aa16c",
            "https://images.unsplash.com/photo-1599838082471-71ca8597304e"
        ],
        'kids_glasses': [
            "https://images.pexels.com/photos/1912868/pexels-photo-1912868.jpeg",
            "https://images.pexels.com/photos/1068205/pexels-photo-1068205.jpeg",
            "https://images.pexels.com/photos/1068207/pexels-photo-1068207.jpeg",
            "https://images.unsplash.com/photo-1588413453099-f1acf8a4a37c",
            "https://images.unsplash.com/photo-1588392382834-a891154bca4d"
        ]
    }

    for source in image_sources:
        if source not in search_terms:
            print(f"Warning: Unknown source '{source}'. Skipping.")
            continue

        terms = search_terms[source]
        source_dir = os.path.join(raw_dir, source)
        os.makedirs(source_dir, exist_ok=True)

        # Number of images to scrape for this source
        source_num = min(num_images // len(image_sources), 20)  # Limit to 20 per category

        # Try to use Pexels API if key is provided
        if pexels_api_key and pexels_api_key != "YOUR_PEXELS_API_KEY":
            try:
                for i in tqdm(range(source_num), desc=f"Downloading {source} images"):
                    term = random.choice(terms)

                    # Pexels API request
                    headers = {
                        "Authorization": pexels_api_key
                    }
                    response = requests.get(
                        f"https://api.pexels.com/v1/search?query={term}&per_page=1&page={i+1}",
                        headers=headers
                    )

                    if response.status_code == 200:
                        data = response.json()
                        if data["photos"]:
                            photo = data["photos"][0]
                            img_url = photo["src"]["large"]

                            # Download image
                            img_response = requests.get(img_url, stream=True)
                            if img_response.status_code == 200:
                                img = Image.open(img_response.raw)
                                img_path = os.path.join(source_dir, f"{source}_{i:04d}.jpg")
                                img.save(img_path)

                                # Add to metadata
                                metadata.append({
                                    'id': f"{source}_{i:04d}",
                                    'path': img_path,
                                    'source': source,
                                    'search_term': term,
                                    'type': source,
                                    'url': img_url
                                })
                            else:
                                print(f"Failed to download image: {img_url}")
                        else:
                            print(f"No photos found for term: {term}")
                    else:
                        print(f"API request failed with status code: {response.status_code}")
                        break

            except Exception as e:
                print(f"Error using Pexels API: {e}")
                print("Falling back to pre-selected images...")
                # Fall back to pre-selected images
                pexels_api_key = None

        # If API key is not provided or API failed, use fallback URLs
        if not pexels_api_key or pexels_api_key == "YOUR_PEXELS_API_KEY":
            urls = fallback_urls[source]
            for i, url in enumerate(urls):
                if i >= source_num:
                    break

                try:
                    # Download image
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        img = Image.open(response.raw)
                        img_path = os.path.join(source_dir, f"{source}_{i:04d}.jpg")
                        img.save(img_path)

                        # Add to metadata
                        metadata.append({
                            'id': f"{source}_{i:04d}",
                            'path': img_path,
                            'source': source,
                            'search_term': source,
                            'type': source,
                            'url': url
                        })
                    else:
                        print(f"Failed to download image: {url}")
                except Exception as e:
                    print(f"Error downloading image {url}: {e}")

    # Save metadata
    with open(os.path.join(data_root, 'raw_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Downloaded {len(metadata)} images.")
    print(f"Images saved to {raw_dir}")
    print(f"Metadata saved to {os.path.join(data_root, 'raw_metadata.json')}")

def process_images(data_root, batch_size=50):
    """
    Process and normalize the collected images

    Args:
        data_root: Root directory for data
        batch_size: Batch size for processing large datasets
    """
    raw_dir = os.path.join(data_root, 'raw_images')
    processed_dir = os.path.join(data_root, 'processed_images')
    os.makedirs(processed_dir, exist_ok=True)

    # Try to load metadata if it exists
    metadata = []
    try:
        with open(os.path.join(data_root, 'raw_metadata.json'), 'r') as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} items from metadata file")
    except FileNotFoundError:
        print("Metadata file not found. Scanning raw images directory...")

        # Scan raw images directory
        for category_dir in os.listdir(raw_dir):
            category_path = os.path.join(raw_dir, category_dir)
            if os.path.isdir(category_path):
                for img_file in os.listdir(category_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(category_path, img_file)
                        img_id = os.path.splitext(img_file)[0]
                        metadata.append({
                            'id': img_id,
                            'path': img_path,
                            'source': category_dir,
                            'type': category_dir
                        })

        print(f"Found {len(metadata)} images in raw directory")

        # Save raw metadata
        with open(os.path.join(data_root, 'raw_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    # Process images in batches
    processed_metadata = []
    total_batches = (len(metadata) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(metadata))
        batch = metadata[start_idx:end_idx]

        print(f"Processing batch {batch_idx+1}/{total_batches} ({len(batch)} images)")

        for item in tqdm(batch, desc=f"Batch {batch_idx+1}"):
            try:
                # Load image
                img_path = item['path']
                img = Image.open(img_path).convert('RGB')

                # Resize to a standard size
                img = img.resize((512, 512), Image.LANCZOS)

                # Apply advanced image processing
                # 1. Normalize brightness and contrast
                img_array = np.array(img)

                # Simple normalization (in a real implementation, use more sophisticated techniques)
                img_array = np.clip((img_array - img_array.min()) * (255.0 / (img_array.max() - img_array.min())), 0, 255).astype(np.uint8)

                # 2. Enhance edges for better feature detection
                try:
                    from scipy import ndimage
                    img_array = ndimage.gaussian_filter(img_array, sigma=0.5)
                except ImportError:
                    pass  # Skip if scipy not available

                # Convert back to PIL Image
                img = Image.fromarray(img_array)

                # Save processed image
                processed_path = os.path.join(processed_dir, f"{item['id']}.jpg")
                img.save(processed_path)

                # Update metadata
                item['processed_path'] = processed_path
                processed_metadata.append(item)

            except Exception as e:
                print(f"Error processing {item.get('id', 'unknown')}: {e}")

        # Save processed metadata after each batch (for recovery in case of failure)
        with open(os.path.join(data_root, 'processed_metadata.json'), 'w') as f:
            json.dump(processed_metadata, f, indent=2)

        print(f"Completed batch {batch_idx+1}/{total_batches}")

    print(f"Processed {len(processed_metadata)} images.")
    print(f"Processed images saved to {processed_dir}")
    print(f"Metadata saved to {os.path.join(data_root, 'processed_metadata.json')}")

def augment_data(data_root, batch_size=50):
    """
    Augment the existing data with variations

    Args:
        data_root: Root directory for data
        batch_size: Batch size for processing large datasets
    """
    processed_dir = os.path.join(data_root, 'processed_images')
    augmented_dir = os.path.join(data_root, 'augmented_images')
    os.makedirs(augmented_dir, exist_ok=True)

    # Load metadata
    try:
        with open(os.path.join(data_root, 'processed_metadata.json'), 'r') as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} items from processed metadata")
    except FileNotFoundError:
        print("Processed metadata file not found. Please run process_images first.")
        return

    # Check if augmented metadata already exists
    try:
        with open(os.path.join(data_root, 'augmented_metadata.json'), 'r') as f:
            augmented_metadata = json.load(f)
        print(f"Loaded {len(augmented_metadata)} items from existing augmented metadata")

        # Check which items have already been augmented
        augmented_ids = set(item['original_id'] for item in augmented_metadata if 'original_id' in item)
        metadata = [item for item in metadata if item['id'] not in augmented_ids]
        print(f"Found {len(metadata)} items that need augmentation")

    except FileNotFoundError:
        # Start with original images
        augmented_metadata = metadata.copy()
        print(f"Starting new augmentation with {len(augmented_metadata)} original images")

    # Define augmentations - expanded set for more variety
    augmentations = [
        ('rotate_5', lambda img: img.rotate(5)),
        ('rotate_-5', lambda img: img.rotate(-5)),
        ('rotate_10', lambda img: img.rotate(10)),
        ('rotate_-10', lambda img: img.rotate(-10)),
        ('brightness_up', lambda img: Image.fromarray(np.clip(np.array(img) * 1.2, 0, 255).astype(np.uint8))),
        ('brightness_down', lambda img: Image.fromarray(np.clip(np.array(img) * 0.8, 0, 255).astype(np.uint8))),
        ('contrast_up', lambda img: Image.fromarray(np.clip((np.array(img) - 128) * 1.2 + 128, 0, 255).astype(np.uint8))),
        ('contrast_down', lambda img: Image.fromarray(np.clip((np.array(img) - 128) * 0.8 + 128, 0, 255).astype(np.uint8))),
        ('flip', lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
        ('blur', lambda img: img.filter(ImageFilter.GaussianBlur(radius=1))),
        ('sharpen', lambda img: img.filter(ImageFilter.SHARPEN)),
        ('crop_center', lambda img: img.crop((img.width * 0.1, img.height * 0.1, img.width * 0.9, img.height * 0.9)).resize((img.width, img.height))),
    ]

    # Process in batches
    total_batches = (len(metadata) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(metadata))
        batch = metadata[start_idx:end_idx]

        print(f"Augmenting batch {batch_idx+1}/{total_batches} ({len(batch)} images)")

        for item in tqdm(batch, desc=f"Batch {batch_idx+1}"):
            try:
                # Load image
                img_path = item.get('processed_path', item.get('path'))
                if not img_path:
                    print(f"No image path found for {item.get('id', 'unknown')}")
                    continue

                img = Image.open(img_path).convert('RGB')

                # Apply augmentations
                for aug_name, aug_func in augmentations:
                    try:
                        # Apply augmentation
                        aug_img = aug_func(img)

                        # Save augmented image
                        aug_id = f"{item['id']}_{aug_name}"
                        aug_path = os.path.join(augmented_dir, f"{aug_id}.jpg")
                        aug_img.save(aug_path)

                        # Add to metadata
                        aug_item = item.copy()
                        aug_item['id'] = aug_id
                        aug_item['augmented_path'] = aug_path
                        aug_item['original_id'] = item['id']
                        aug_item['augmentation'] = aug_name
                        augmented_metadata.append(aug_item)
                    except Exception as e:
                        print(f"Error applying augmentation {aug_name} to {item.get('id', 'unknown')}: {e}")

            except Exception as e:
                print(f"Error augmenting {item.get('id', 'unknown')}: {e}")

        # Save augmented metadata after each batch
        with open(os.path.join(data_root, 'augmented_metadata.json'), 'w') as f:
            json.dump(augmented_metadata, f, indent=2)

        print(f"Completed batch {batch_idx+1}/{total_batches}")

    # Calculate statistics
    original_count = len([item for item in augmented_metadata if 'augmentation' not in item])
    augmented_count = len(augmented_metadata) - original_count

    print(f"Dataset contains {original_count} original images and {augmented_count} augmented images.")
    print(f"Total dataset size: {len(augmented_metadata)} images.")
    print(f"Augmented images saved to {augmented_dir}")
    print(f"Metadata saved to {os.path.join(data_root, 'augmented_metadata.json')}")

def split_data(data_root, train_ratio, val_ratio):
    """Split the data into train/val/test sets"""
    # Load metadata
    try:
        with open(os.path.join(data_root, 'augmented_metadata.json'), 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        try:
            with open(os.path.join(data_root, 'processed_metadata.json'), 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print("No metadata file found. Please run process_images or augment_data first.")
            return

    # Shuffle the data
    random.shuffle(metadata)

    # Calculate split indices
    n = len(metadata)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split the data
    train_data = metadata[:train_end]
    val_data = metadata[train_end:val_end]
    test_data = metadata[val_end:]

    # Create directories
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'meshes', 'textures']:
            os.makedirs(os.path.join(data_root, split, subdir), exist_ok=True)

    # Process each split
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split, data in splits.items():
        split_metadata = []

        for item in tqdm(data, desc=f"Processing {split} split"):
            # Get the image path (prefer augmented if available)
            if 'augmented_path' in item:
                src_path = item['augmented_path']
            elif 'processed_path' in item:
                src_path = item['processed_path']
            else:
                src_path = item['path']

            # Copy image to split directory
            dst_path = os.path.join(data_root, split, 'images', f"{item['id']}.jpg")
            shutil.copy(src_path, dst_path)

            # Add to split metadata
            split_metadata.append({
                'id': item['id'],
                'image_path': f"{split}/images/{item['id']}.jpg",
                'mesh_path': f"{split}/meshes/{item['id']}.obj",  # Placeholder
                'texture_path': f"{split}/textures/{item['id']}.png",  # Placeholder
                'type': item.get('type', 'unknown')
            })

        # Save split metadata
        with open(os.path.join(data_root, f"{split}_metadata.json"), 'w') as f:
            json.dump(split_metadata, f, indent=2)

        print(f"Processed {len(split_metadata)} images for {split} split.")

    print(f"Data split into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test samples.")

def download_large_dataset(data_root, dataset_source, dataset_url=None, batch_size=50):
    """
    Download a large pre-existing dataset of glasses images

    Args:
        data_root: Root directory for data
        dataset_source: Source of the dataset (kaggle, google, custom_url)
        dataset_url: URL for custom dataset download
        batch_size: Batch size for processing
    """
    print(f"Downloading large dataset from {dataset_source}...")

    # Create raw directory
    raw_dir = os.path.join(data_root, 'raw_images')
    os.makedirs(raw_dir, exist_ok=True)

    # Different dataset sources
    if dataset_source == 'kaggle':
        try:
            # Check if kaggle is installed
            import importlib
            kaggle_spec = importlib.util.find_spec("kaggle")
            if kaggle_spec is None:
                print("Kaggle API not found. Installing...")
                import subprocess
                subprocess.check_call(["pip", "install", "kaggle"])

            import kaggle

            # List of relevant Kaggle datasets for glasses
            kaggle_datasets = [
                "luxonis/glasses-detection-dataset",
                "tapakah68/glasses-and-headwear-detection",
                "tapakah68/eyeglasses-detection-dataset"
            ]

            for dataset in kaggle_datasets:
                try:
                    print(f"Downloading Kaggle dataset: {dataset}")
                    kaggle.api.dataset_download_files(dataset, path=os.path.join(raw_dir, 'kaggle'), unzip=True)
                except Exception as e:
                    print(f"Error downloading {dataset}: {e}")

            # Process downloaded files
            process_kaggle_datasets(data_root, os.path.join(raw_dir, 'kaggle'), batch_size)

        except Exception as e:
            print(f"Error using Kaggle API: {e}")
            print("Please make sure you have set up your Kaggle API credentials.")
            print("See https://github.com/Kaggle/kaggle-api for instructions.")

    elif dataset_source == 'google':
        try:
            # Use Google Dataset Search API (requires API key)
            print("Downloading from Google Dataset Search...")
            print("This feature requires a Google API key.")

            # Mock implementation - in a real scenario, you would use the Google Dataset Search API
            print("Mock implementation: Creating sample data...")

            # Create sample directories
            for category in ['eyeglasses', 'sunglasses', 'reading_glasses', 'sports_glasses']:
                category_dir = os.path.join(raw_dir, category)
                os.makedirs(category_dir, exist_ok=True)

                # Create sample images
                for i in range(100):  # 100 images per category
                    img = Image.new('RGB', (512, 512), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                    img_path = os.path.join(category_dir, f"{category}_{i:04d}.jpg")
                    img.save(img_path)

        except Exception as e:
            print(f"Error downloading from Google Dataset Search: {e}")

    elif dataset_source == 'custom_url' and dataset_url:
        try:
            import zipfile
            import io

            print(f"Downloading dataset from {dataset_url}...")

            # Download the dataset
            response = requests.get(dataset_url)
            if response.status_code == 200:
                # Check if it's a zip file
                if dataset_url.endswith('.zip'):
                    # Extract the zip file
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                        zip_ref.extractall(raw_dir)
                    print(f"Extracted zip file to {raw_dir}")
                else:
                    # Save the file
                    file_name = dataset_url.split('/')[-1]
                    with open(os.path.join(raw_dir, file_name), 'wb') as f:
                        f.write(response.content)
                    print(f"Downloaded {file_name} to {raw_dir}")
            else:
                print(f"Failed to download dataset: HTTP {response.status_code}")

        except Exception as e:
            print(f"Error downloading from custom URL: {e}")

    else:
        print(f"Unknown dataset source: {dataset_source}")
        print("Available options: kaggle, google, custom_url")
        return

    print(f"Dataset download completed. Files saved to {raw_dir}")

def process_kaggle_datasets(data_root, kaggle_dir, batch_size=50):
    """Process Kaggle datasets and organize them"""
    print("Processing Kaggle datasets...")

    # Create category directories
    categories = ['eyeglasses', 'sunglasses', 'reading_glasses', 'sports_glasses']
    for category in categories:
        os.makedirs(os.path.join(data_root, 'raw_images', category), exist_ok=True)

    # Find all image files
    image_files = []
    for root, _, files in os.walk(kaggle_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} images in Kaggle datasets")

    # Process in batches
    for i in range(0, len(image_files), batch_size):
        batch = image_files[i:i+batch_size]

        for img_path in tqdm(batch, desc=f"Processing batch {i//batch_size + 1}/{len(image_files)//batch_size + 1}"):
            try:
                # Determine category based on filename or path
                file_name = os.path.basename(img_path).lower()

                if 'sun' in file_name or 'sun' in img_path.lower():
                    category = 'sunglasses'
                elif 'read' in file_name or 'read' in img_path.lower():
                    category = 'reading_glasses'
                elif 'sport' in file_name or 'sport' in img_path.lower() or 'goggles' in file_name:
                    category = 'sports_glasses'
                else:
                    category = 'eyeglasses'  # Default category

                # Copy to appropriate category
                dst_dir = os.path.join(data_root, 'raw_images', category)
                dst_path = os.path.join(dst_dir, f"{category}_{len(os.listdir(dst_dir)):04d}.jpg")

                # Open, convert, and save the image
                img = Image.open(img_path).convert('RGB')
                img.save(dst_path)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print("Kaggle dataset processing completed")

def main():
    args = parse_args()

    # Download large dataset if requested
    if args.download_dataset:
        download_large_dataset(args.data_root, args.download_dataset, args.dataset_url, args.batch_size)

    if args.scrape_images:
        scrape_images(args.data_root, args.num_images, args.image_sources)

    if args.process_images:
        process_images(args.data_root)

    if args.augment_data:
        augment_data(args.data_root)

    if args.split_data:
        split_data(args.data_root, args.train_ratio, args.val_ratio)

    print("Data collection and preparation completed!")

if __name__ == "__main__":
    main()
