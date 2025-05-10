import os
import argparse
import json
import shutil
import requests
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Data collection and preparation for glasses 3D reconstruction')
    parser.add_argument('--data_root', type=str, default='data/glasses', help='Root directory for data')
    parser.add_argument('--scrape_images', action='store_true', help='Scrape glasses images from the web')
    parser.add_argument('--process_images', action='store_true', help='Process and normalize collected images')
    parser.add_argument('--augment_data', action='store_true', help='Augment existing data')
    parser.add_argument('--split_data', action='store_true', help='Split data into train/val/test')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation data')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to scrape')
    parser.add_argument('--image_sources', type=str, nargs='+',
                        default=['eyeglasses', 'sunglasses', 'reading_glasses', 'sports_glasses'],
                        help='Types of glasses to scrape')
    return parser.parse_args()

def scrape_images(data_root, num_images, image_sources):
    """
    Scrape glasses images from various sources using Unsplash API
    """
    raw_dir = os.path.join(data_root, 'raw_images')
    os.makedirs(raw_dir, exist_ok=True)

    # Search terms for different types of glasses
    search_terms = {
        'eyeglasses': ['eyeglasses', 'prescription glasses', 'optical frames'],
        'sunglasses': ['sunglasses', 'shades', 'sun glasses'],
        'reading_glasses': ['reading glasses', 'computer glasses'],
        'sports_glasses': ['sports glasses', 'athletic eyewear', 'sports goggles']
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

def process_images(data_root):
    """Process and normalize the collected images"""
    raw_dir = os.path.join(data_root, 'raw_images')
    processed_dir = os.path.join(data_root, 'processed_images')
    os.makedirs(processed_dir, exist_ok=True)

    # Load metadata
    try:
        with open(os.path.join(data_root, 'raw_metadata.json'), 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Metadata file not found. Please run scrape_images first.")
        return

    processed_metadata = []

    for item in tqdm(metadata, desc="Processing images"):
        try:
            # Load image
            img_path = item['path']
            img = Image.open(img_path).convert('RGB')

            # Resize to a standard size
            img = img.resize((512, 512), Image.LANCZOS)

            # Normalize brightness and contrast
            # In a real implementation, you would use more sophisticated techniques

            # Save processed image
            processed_path = os.path.join(processed_dir, f"{item['id']}.jpg")
            img.save(processed_path)

            # Update metadata
            item['processed_path'] = processed_path
            processed_metadata.append(item)

        except Exception as e:
            print(f"Error processing {item['id']}: {e}")

    # Save processed metadata
    with open(os.path.join(data_root, 'processed_metadata.json'), 'w') as f:
        json.dump(processed_metadata, f, indent=2)

    print(f"Processed {len(processed_metadata)} images.")
    print(f"Processed images saved to {processed_dir}")
    print(f"Metadata saved to {os.path.join(data_root, 'processed_metadata.json')}")

def augment_data(data_root):
    """Augment the existing data with variations"""
    processed_dir = os.path.join(data_root, 'processed_images')
    augmented_dir = os.path.join(data_root, 'augmented_images')
    os.makedirs(augmented_dir, exist_ok=True)

    # Load metadata
    try:
        with open(os.path.join(data_root, 'processed_metadata.json'), 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Processed metadata file not found. Please run process_images first.")
        return

    augmented_metadata = []

    # Add original images to augmented metadata
    for item in metadata:
        augmented_metadata.append(item)

    # Define augmentations
    augmentations = [
        ('rotate_5', lambda img: img.rotate(5)),
        ('rotate_-5', lambda img: img.rotate(-5)),
        ('brightness_up', lambda img: Image.fromarray(np.clip(np.array(img) * 1.2, 0, 255).astype(np.uint8))),
        ('brightness_down', lambda img: Image.fromarray(np.clip(np.array(img) * 0.8, 0, 255).astype(np.uint8))),
        ('flip', lambda img: img.transpose(Image.FLIP_LEFT_RIGHT))
    ]

    for item in tqdm(metadata, desc="Augmenting images"):
        try:
            # Load image
            img_path = item['processed_path']
            img = Image.open(img_path).convert('RGB')

            # Apply augmentations
            for aug_name, aug_func in augmentations:
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
            print(f"Error augmenting {item['id']}: {e}")

    # Save augmented metadata
    with open(os.path.join(data_root, 'augmented_metadata.json'), 'w') as f:
        json.dump(augmented_metadata, f, indent=2)

    print(f"Created {len(augmented_metadata) - len(metadata)} augmented images.")
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

def main():
    args = parse_args()

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
