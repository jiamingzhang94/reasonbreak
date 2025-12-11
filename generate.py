#!/usr/bin/env python3
import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from train import AdversarialPrivacyTrainer, PrivacyProtectionDataset, AdvNoiseTensorProcessor
from models.model import Decoder, CLIPEncoder


class AdversarialGenerator:
    def __init__(self, eps: float, max_num_blocks: int, decoder_path: str, embedding_bank_path: str,
                 device: str = 'cuda', save_format: str = 'jpg', jpg_quality: int = 100, image_root: str = None):
        self.device = device
        self.epsilon = eps
        self.image_root = image_root

        print("🚀 Initializing lightweight adversarial generator (no surrogate models)...")
        if self.image_root:
            print(f"📂 Image Root Directory set to: {self.image_root}")

        print("Loading CLIP encoder for embedding bank...")
        self.clip_encoder = CLIPEncoder("ViT-B/32").to(device)
        self.clip_encoder.eval()

        print(f"Loading embedding bank from {embedding_bank_path}")
        self.embedding_bank = torch.load(embedding_bank_path, map_location=device)
        print(f"Embedding bank shape: {self.embedding_bank.shape}")

        print(f"Loading trained decoder from {decoder_path}")
        self.decoder = Decoder(embed_dim=512, img_channels=3, img_size=224).to(device)
        checkpoint = torch.load(decoder_path, map_location=device)

        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.decoder.load_state_dict(checkpoint)

        self.decoder.eval()

        self.tensor_processor = AdvNoiseTensorProcessor(debug=False)
        self.max_num_blocks = max_num_blocks
        self.save_format = save_format.lower()
        self.jpg_quality = jpg_quality

        print("✅ Lightweight generator initialized successfully (no surrogate models loaded)")

    def parse_reasoning_chain(self, gemini_response: str) -> List[Dict]:
        try:
            start_idx = gemini_response.find('{')
            end_idx = gemini_response.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                print("Warning: No JSON found in gemini_response")
                return []

            json_str = gemini_response[start_idx:end_idx]
            data = json.loads(json_str)

            if 'reasoning_chain' not in data:
                print("Warning: No reasoning_chain found in JSON")
                return []

            reasoning_chain = data['reasoning_chain']

            for step in reasoning_chain:
                if 'square_bbox' in step:
                    bbox = step['square_bbox']
                    if isinstance(bbox, list) and len(bbox) == 3:
                        continue
                    elif isinstance(bbox, list) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        size = max(width, height)
                        step['square_bbox'] = [center_x, center_y, size]
                    else:
                        print(f"Warning: Unexpected square_bbox format: {bbox}")
                        step['square_bbox'] = [0.5, 0.5, 0.3]
                elif 'bbox' in step:
                    bbox = step['bbox']
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        size = max(width, height)
                        step['square_bbox'] = [center_x, center_y, size]
                    else:
                        print(f"Warning: Unexpected bbox format: {bbox}")
                        step['square_bbox'] = [0.5, 0.5, 0.3]
                else:
                    print("Warning: No bbox or square_bbox found in step")
                    step['square_bbox'] = [0.5, 0.5, 0.3]

            return reasoning_chain

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return []
        except Exception as e:
            print(f"Error processing reasoning chain: {e}")
            return []

    def get_target_embeddings(self, key_concepts_batch: List[List[str]]) -> torch.Tensor:
        target_embeddings = []

        with torch.no_grad():
            for i, concepts in enumerate(key_concepts_batch):
                if not concepts:
                    raise ValueError(
                        f"Block {i} has no key concepts. This should not happen - check your data preprocessing.")

                text_embeddings = self.clip_encoder.encode_text(concepts, self.device)
                text_embeddings = F.normalize(text_embeddings, dim=-1)

                if len(concepts) == 1:
                    similarities = F.cosine_similarity(
                        text_embeddings, self.embedding_bank.to(self.device), dim=-1
                    )
                    farthest_idx = similarities.argmin()
                else:
                    all_similarities = []
                    for j in range(len(concepts)):
                        sim = F.cosine_similarity(
                            text_embeddings[j:j + 1], self.embedding_bank.to(self.device), dim=-1
                        )
                        all_similarities.append(sim)

                    max_similarities = torch.stack(all_similarities).max(dim=0)[0]
                    farthest_idx = max_similarities.argmin()

                target_embeddings.append(self.embedding_bank[farthest_idx:farthest_idx + 1])

        result = torch.cat(target_embeddings, dim=0).to(self.device)

        assert result.shape[0] == len(
            key_concepts_batch), f"result shape[0] ({result.shape[0]}) must match input length ({len(key_concepts_batch)})"
        assert result.shape[1] == 512, f"result shape[1] must be 512, got {result.shape[1]}"

        return result

    def generate_adversarial_sample(self, image_path: str, reasoning_chain: List[Dict]) -> np.ndarray:
        epsilon = self.epsilon
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
                temp_file.write('{"image_path": "dummy", "location_analysis": {"reasoning_chain": []}}\n')
                temp_jsonl_path = temp_file.name

            processor = PrivacyProtectionDataset(temp_jsonl_path, debug=False, max_num=self.max_num_blocks)

            import os
            os.unlink(temp_jsonl_path)

            image = Image.open(image_path).convert('RGB')
            orig_width, orig_height = image.size

            processed_images, target_aspect_ratio = processor.dynamic_preprocess_with_info(image)

            pixel_values = torch.stack([processor.transform(img) for img in processed_images])
            pixel_values = pixel_values.to(self.device)

            key_concepts = processor.assign_key_concepts(
                reasoning_chain, target_aspect_ratio, orig_width, orig_height
            )

            target_embeddings = self.get_target_embeddings(key_concepts)

            with torch.no_grad():
                raw_noise_blocks = self.decoder(target_embeddings)
                noise_blocks = torch.clamp(raw_noise_blocks, -epsilon, epsilon)

                reconstructed_noise = self.tensor_processor.reconstruct_image(
                    noise_blocks, target_aspect_ratio, (orig_width, orig_height)
                )

                import torchvision.transforms as T
                original_tensor = T.ToTensor()(image).to(self.device)

                if reconstructed_noise.shape != original_tensor.shape:
                    reconstructed_noise = F.interpolate(
                        reconstructed_noise.unsqueeze(0),
                        size=original_tensor.shape[1:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                adversarial_tensor = original_tensor + reconstructed_noise
                adversarial_tensor = torch.clamp(adversarial_tensor, 0, 1)

                adversarial_image = T.ToPILImage()(adversarial_tensor.cpu())

                return np.array(adversarial_image)

        except Exception as e:
            print(f"Error generating adversarial sample for {image_path}: {e}")
            import traceback
            traceback.print_exc()
            try:
                image = Image.open(image_path).convert('RGB')
                return np.array(image)
            except:
                return None

    def process_json_file(self, json_path: str, output_dir: str) -> List[Dict]:
        print(f"Processing {json_path}")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            results = []

            for item in tqdm(data, desc=f"Processing {Path(json_path).name}"):
                try:
                    raw_filename = item['filename']
                    
                    if self.image_root:
                        rel_path = raw_filename.lstrip(os.sep)
                        full_filename = os.path.join(self.image_root, rel_path)
                    else:
                        full_filename = raw_filename

                    ground_truth = item['ground_truth']
                    gemini_response = item['gemini_response']

                    if not os.path.exists(full_filename):
                        print(f"Warning: Image not found: {full_filename}")
                        if self.image_root:
                            print(f"  Checked path: {full_filename}")
                            print(f"  Based on root: {self.image_root}")
                        continue

                    reasoning_chain = self.parse_reasoning_chain(gemini_response)

                    if not reasoning_chain:
                        print(f"Warning: No valid reasoning chain for {full_filename}")
                        continue

                    adversarial_image = self.generate_adversarial_sample(full_filename, reasoning_chain)
                    
                    if adversarial_image is None:
                        continue

                    original_path_obj = Path(full_filename)
                    if self.save_format == 'png':
                        output_filename = original_path_obj.stem + '.png'
                        output_path = Path(output_dir) / output_filename

                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        adversarial_pil = Image.fromarray(adversarial_image)
                        adversarial_pil.save(output_path, format='PNG')
                    else:
                        output_filename = original_path_obj.stem + '.jpg'
                        output_path = Path(output_dir) / output_filename

                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        adversarial_pil = Image.fromarray(adversarial_image)
                        adversarial_pil.save(output_path, format='JPEG', quality=self.jpg_quality, optimize=False)

                    result = {
                        'filename': str(output_path),
                        'address': str(ground_truth.get('address', '')),
                        'latitude': str(ground_truth.get('latitude', '')),
                        'longitude': str(ground_truth.get('longitude', ''))
                    }
                    results.append(result)

                    print(f"✅ Generated: {output_path}")

                except Exception as e:
                    print(f"Error processing item {item.get('filename', 'unknown')}: {e}")
                    continue

            return results

        except Exception as e:
            print(f"Error reading JSON file {json_path}: {e}")
            return []

    def generate_from_json(self, json_path: str, output_dir: str, csv_path: str = None):
        print("🚀 Starting adversarial sample generation...")

        os.makedirs(output_dir, exist_ok=True)

        all_results = []

        if os.path.isfile(json_path):
            results = self.process_json_file(json_path, output_dir)
            all_results.extend(results)
        elif os.path.isdir(json_path):
            json_files = list(Path(json_path).glob("*.json"))
            if not json_files:
                print(f"No JSON files found in {json_path}")
                return

            for json_file in json_files:
                results = self.process_json_file(str(json_file), output_dir)
                all_results.extend(results)
        else:
            print(f"Invalid path: {json_path}")
            return

        if all_results:
            if csv_path is None:
                csv_path = os.path.join(output_dir, 'annotations.csv')

            df = pd.DataFrame(all_results)
            df.to_csv(csv_path, index=False)
            print(f"📊 CSV annotations saved to: {csv_path}")
            print(f"Generated {len(all_results)} adversarial samples")
        else:
            print("❌ No adversarial samples were generated")

        print("✅ Generation completed!")


def main():
    parser = argparse.ArgumentParser(description='Generate adversarial samples from JSON data')
    parser.add_argument('--decoder_path', type=str)
    parser.add_argument('--embedding_bank_path', type=str)
    parser.add_argument('--json_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--image_root', type=str)
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--epsilon', type=float, default=0.0627)
    parser.add_argument('--max_num_blocks', type=int, default=64)
    parser.add_argument('--save_format', type=str, default='jpg', choices=['jpg', 'png'])
    parser.add_argument('--jpg_quality', type=int, default=95)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    generator = AdversarialGenerator(
        eps=args.epsilon,
        max_num_blocks=args.max_num_blocks,
        decoder_path=args.decoder_path,
        embedding_bank_path=args.embedding_bank_path,
        device=args.device,
        save_format=args.save_format,
        jpg_quality=args.jpg_quality,
        image_root=args.image_root
    )

    generator.generate_from_json(
        json_path=args.json_path,
        output_dir=args.output_dir,
        csv_path=args.csv_path
    )


if __name__ == "__main__":
    main()