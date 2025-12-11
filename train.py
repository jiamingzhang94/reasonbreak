import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import argparse
from models.model import Decoder, CLIPEncoder
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor, 
    BlipForImageTextRetrieval,
    SiglipProcessor,
    SiglipModel,
    AlignProcessor,
    AlignModel,
    AltCLIPProcessor,
    AltCLIPModel
)

if torch.cuda.is_available():
    torch.cuda.init()

Image.MAX_IMAGE_PIXELS = 200000000

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
NOISE_IMAGE_SIZE = 224

SURROGATE_MODEL_CONFIGS = {
    "openai/clip-vit-base-patch32": {"type": "clip"},
    "openai/clip-vit-base-patch16": {"type": "clip"},
    "openai/clip-vit-large-patch14": {"type": "clip"},
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": {"type": "clip"},
    "laion/CLIP-ViT-B-16-laion2B-s34B-b88K": {"type": "clip"},
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": {"type": "clip"},
    "google/siglip-base-patch16-224": {"type": "siglip"},
    "google/siglip-large-patch16-256": {"type": "siglip"},
    "Salesforce/blip-image-captioning-base": {"type": "blip"},
    "Salesforce/blip-image-captioning-large": {"type": "blip"},
    "kakaobrain/align-base": {"type": "align"},
    "BAAI/AltCLIP-XLMR-L": {"type": "altclip"},
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {"type": "clip"},
    "laion/CLIP-ViT-g-14-laion2B-s12B-b42K": {"type": "clip"},
    "ViT-B/32": {"type": "clip", "hf_name": "openai/clip-vit-base-patch32"},
    "ViT-B/16": {"type": "clip", "hf_name": "openai/clip-vit-base-patch16"}, 
    "ViT-L/14": {"type": "clip", "hf_name": "openai/clip-vit-large-patch14"},
    "RN50": {"type": "clip", "hf_name": "openai/clip-vit-base-patch32"},
    "RN101": {"type": "clip", "hf_name": "openai/clip-vit-base-patch16"},
}

class SurrogateModel:
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device
        
        if model_name not in SURROGATE_MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}")
        
        config = SURROGATE_MODEL_CONFIGS[model_name]
        self.model_type = config["type"]
        actual_name = config.get("hf_name", model_name)
        
        print(f"Loading surrogate model: {actual_name} (type: {self.model_type})")
        
        if self.model_type == "clip":
            self.model = CLIPModel.from_pretrained(actual_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(actual_name)
        elif self.model_type == "siglip":
            self.model = SiglipModel.from_pretrained(actual_name).to(device)
            self.processor = SiglipProcessor.from_pretrained(actual_name)
        elif self.model_type == "blip":
            self.model = BlipForImageTextRetrieval.from_pretrained(actual_name).to(device)
            self.processor = BlipProcessor.from_pretrained(actual_name)
        elif self.model_type == "align":
            self.model = AlignModel.from_pretrained(actual_name).to(device)
            self.processor = AlignProcessor.from_pretrained(actual_name)
        elif self.model_type == "altclip":
            self.model = AltCLIPModel.from_pretrained(actual_name).to(device)
            self.processor = AltCLIPProcessor.from_pretrained(actual_name)
        
        self.model.eval()
        
    def encode_image(self, images: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        if self.model_type in ["clip", "siglip", "blip", "align", "altclip"]:
            batch_size = images.shape[0]
            pil_images = []
            for i in range(batch_size):
                if requires_grad:
                    img_tensor = images[i]
                else:
                    img_array = (images[i].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_array)
                    pil_images.append(pil_img)
            
            if requires_grad:
                mean = torch.tensor([0.485, 0.456, 0.406]).to(images.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).to(images.device).view(1, 3, 1, 1)
                normalized_images = (images - mean) / std
                
                if normalized_images.shape[-1] != 224:
                    normalized_images = F.interpolate(normalized_images, size=(224, 224), mode='bilinear', align_corners=False)
                
                image_features = self.model.get_image_features(pixel_values=normalized_images)
                return image_features
            else:
                with torch.no_grad():
                    inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    image_features = self.model.get_image_features(**inputs)
                    return image_features
            
        raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def encode_text(self, texts: list) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            return text_features

class PrivacyProtectionDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str = '', image_size: int = NOISE_IMAGE_SIZE, max_num: int = 12, debug: bool = False):
        self.data = []
        self.image_size = image_size
        self.max_num = max_num
        self.debug = debug
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if image_root and 'image_path' in item:
                    item['image_path'] = os.path.join(image_root, item['image_path'])
                self.data.append(item)
        
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        location_analysis = item['location_analysis']
        
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        processed_images, target_aspect_ratio = self.dynamic_preprocess_with_info(image)
        pixel_values = torch.stack([self.transform(img) for img in processed_images])
        
        key_concepts = self.assign_key_concepts(
            location_analysis['reasoning_chain'], 
            target_aspect_ratio, 
            orig_width, 
            orig_height
        )
        
        return {
            'pixel_values': pixel_values,
            'key_concepts': key_concepts,
            'target_aspect_ratio': target_aspect_ratio,
            'orig_size': (orig_width, orig_height),
            'image_path': image_path
        }

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * self.image_size * self.image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess_with_info(self, image):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set(
            (i, j) for n in range(1, self.max_num + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= self.max_num and i * j >= 1
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height
        )
        
        target_width = self.image_size * target_aspect_ratio[0]
        target_height = self.image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        resized_img = image.resize((target_width, target_height))
        
        processed_images = []
        for i in range(blocks):
            box = (
                (i % target_aspect_ratio[0]) * self.image_size,
                (i // target_aspect_ratio[0]) * self.image_size,
                ((i % target_aspect_ratio[0]) + 1) * self.image_size,
                ((i // target_aspect_ratio[0]) + 1) * self.image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        return processed_images, target_aspect_ratio

    def assign_key_concepts(self, reasoning_chain, target_aspect_ratio, orig_width, orig_height):
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        key_concepts = [[] for _ in range(blocks)]
        
        target_width = self.image_size * target_aspect_ratio[0]
        target_height = self.image_size * target_aspect_ratio[1]
        
        for i in range(blocks):
            block_x = (i % target_aspect_ratio[0]) * self.image_size
            block_y = (i // target_aspect_ratio[0]) * self.image_size
            
            orig_block_x1 = block_x * orig_width / target_width
            orig_block_y1 = block_y * orig_height / target_height
            orig_block_x2 = (block_x + self.image_size) * orig_width / target_width
            orig_block_y2 = (block_y + self.image_size) * orig_height / target_height
            
            has_concept = False
            for step in reasoning_chain:
                if 'square_bbox' not in step:
                    continue
                
                bbox = step['square_bbox']
                
                if not isinstance(bbox, list) or len(bbox) != 3:
                    continue
                
                try:
                    center_x_norm, center_y_norm, size_norm = bbox
                    center_x = float(center_x_norm) * orig_width
                    center_y = float(center_y_norm) * orig_height
                    size = float(size_norm) * min(orig_width, orig_height)
                except (ValueError, TypeError, IndexError):
                    continue
                
                bbox_x1 = center_x - size / 2
                bbox_y1 = center_y - size / 2
                bbox_x2 = center_x + size / 2
                bbox_y2 = center_y + size / 2
                
                if (orig_block_x1 < bbox_x2 and orig_block_x2 > bbox_x1 and 
                    orig_block_y1 < bbox_y2 and orig_block_y2 > bbox_y1):
                    key_concepts[i].append(step['key_concept'])
                    has_concept = True
            
            if not has_concept:
                key_concepts[i] = [step['key_concept'] for step in reasoning_chain]
        
        return key_concepts

def custom_collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    key_concepts = [item['key_concepts'] for item in batch]
    target_aspect_ratios = [item['target_aspect_ratio'] for item in batch]
    orig_sizes = [item['orig_size'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'key_concepts': key_concepts,
        'target_aspect_ratio': target_aspect_ratios,
        'orig_size': orig_sizes,
        'image_path': image_paths
    }

class AdvNoiseTensorProcessor:
    def __init__(self, image_size: int = NOISE_IMAGE_SIZE, debug: bool = False):
        self.image_size = image_size
        self.debug = debug
    
    def reconstruct_image(self, noise_blocks: torch.Tensor, target_aspect_ratio: tuple, 
                         orig_size: tuple) -> torch.Tensor:
        blocks_w, blocks_h = target_aspect_ratio[0], target_aspect_ratio[1]
        device = noise_blocks.device
        
        main_blocks = noise_blocks
        
        target_width = blocks_w * self.image_size
        target_height = blocks_h * self.image_size
        
        reconstructed = torch.zeros(3, target_height, target_width, device=device)
        
        for i in range(len(main_blocks)):
            row = i // blocks_w
            col = i % blocks_w
            
            y1 = row * self.image_size
            y2 = (row + 1) * self.image_size
            x1 = col * self.image_size
            x2 = (col + 1) * self.image_size
            
            reconstructed[:, y1:y2, x1:x2] = main_blocks[i]
        
        orig_width, orig_height = orig_size
        
        final_noise = F.interpolate(
            reconstructed.unsqueeze(0), 
            size=(orig_height, orig_width), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        return final_noise

class AdversarialPrivacyTrainer:
    def __init__(self, embedding_bank_path: str, device: str = 'cuda', lr: float = 1e-4, debug: bool = False,
                 surrogate_models: list = None, use_ensemble: bool = False, use_fp16: bool = False):
        self.device = device
        self.debug = debug
        self.use_ensemble = use_ensemble
        self.use_fp16 = use_fp16
        
        if use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        if not surrogate_models:
            surrogate_models = [
                "openai/clip-vit-base-patch32",
                "openai/clip-vit-base-patch16", 
                "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
            ]
        
        self.surrogate_models = []
        
        if len(surrogate_models) > 1:
            self.use_ensemble = True
        
        for model_name in surrogate_models:
            surrogate = SurrogateModel(model_name, device)
            for param in surrogate.model.parameters():
                param.requires_grad = False
            self.surrogate_models.append(surrogate)
        
        self.clip_encoder = CLIPEncoder("ViT-B/32").to(device)
        self.clip_encoder.eval()
        
        if os.path.exists(embedding_bank_path):
            self.embedding_bank = torch.load(embedding_bank_path, map_location=device)
        else:
            self.embedding_bank = None
        
        self.decoder = Decoder(embed_dim=512, img_channels=3, img_size=NOISE_IMAGE_SIZE).to(device)
        self.tensor_processor = AdvNoiseTensorProcessor(debug=debug)
        self.optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=lr)
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = False):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            self.decoder.load_state_dict(checkpoint)
        
        if resume_training:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_loss = checkpoint.get('loss', float('inf'))
            epsilon = checkpoint.get('epsilon', 8.0/255.0)
            
            return start_epoch, best_loss, epsilon
        else:
            return 0, float('inf'), None
    
    def build_embedding_bank(self, dataset: PrivacyProtectionDataset, save_path: str):
        embeddings = []
        
        simple_transform = T.Compose([
            T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        
        print("Building embedding bank...")
        with torch.no_grad():
            for i in range(len(dataset)):
                if i % 100 == 0:
                    print(f"Processing {i}/{len(dataset)}")
                
                item = dataset.data[i]
                image = Image.open(item['image_path']).convert('RGB')
                image_tensor = simple_transform(image).unsqueeze(0).to(self.device)
                
                image_embedding = self.clip_encoder.encode_img(image_tensor)
                image_embedding = F.normalize(image_embedding, dim=-1)
                embeddings.append(image_embedding.cpu())
        
        embedding_bank = torch.cat(embeddings, dim=0)
        torch.save(embedding_bank, save_path)
        print(f"Embedding bank saved to {save_path}")
        return embedding_bank
    
    def get_target_embeddings(self, key_concepts_batch: list) -> torch.Tensor:
        target_embeddings = []
        
        with torch.no_grad():
            for i, concepts in enumerate(key_concepts_batch):
                if not concepts:
                    raise ValueError(f"Block {i} has no key concepts.")
                
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
                            text_embeddings[j:j+1], self.embedding_bank.to(self.device), dim=-1
                        )
                        all_similarities.append(sim)
                    
                    max_similarities = torch.stack(all_similarities).max(dim=0)[0]
                    farthest_idx = max_similarities.argmin()
                
                target_embeddings.append(self.embedding_bank[farthest_idx:farthest_idx+1])
        
        result = torch.cat(target_embeddings, dim=0).to(self.device)
        return result
    
    def train_step(self, batch, epsilon: float = 8.0/255.0):
        pixel_values = batch['pixel_values'].to(self.device)
        key_concepts = batch['key_concepts']
        
        batch_size = pixel_values.size(0)
        total_loss = 0
        
        from contextlib import nullcontext
        autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16) if self.use_fp16 else nullcontext()
        
        for b in range(batch_size):
            blocks = pixel_values[b]
            concepts = key_concepts[b]
            
            target_embeddings = self.get_target_embeddings(concepts)
            
            with autocast_context:
                raw_noise_blocks = self.decoder(target_embeddings)
                noise_blocks = torch.clamp(raw_noise_blocks, -epsilon, epsilon)
                original_blocks = blocks
                noisy_blocks = original_blocks + noise_blocks
                noisy_blocks = torch.clamp(noisy_blocks, 0, 1)
            
            ensemble_losses = []
            
            for surrogate in self.surrogate_models:
                with autocast_context:
                    original_features = surrogate.encode_image(original_blocks, requires_grad=False)
                    original_features = F.normalize(original_features, dim=-1)
                    
                    noisy_features = surrogate.encode_image(noisy_blocks, requires_grad=True)
                    noisy_features = F.normalize(noisy_features, dim=-1)
                    
                    block_similarities = F.cosine_similarity(original_features, noisy_features, dim=-1)
                    model_loss = block_similarities.mean()
                    ensemble_losses.append(model_loss)
            
            sample_loss = torch.stack(ensemble_losses).mean()
            total_loss += sample_loss
        
        avg_loss = total_loss / batch_size
        
        self.optimizer.zero_grad()
        
        if self.use_fp16 and self.scaler is not None:
            self.scaler.scale(avg_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            avg_loss.backward()
            self.optimizer.step()
        
        return avg_loss.item()
    
    def train(self, dataloader: DataLoader, num_epochs: int = 10, epsilon: float = 8.0/255.0, 
              save_dir: str = './checkpoints', save_interval: int = 1, verbose: bool = False,
              start_epoch: int = 0, best_loss: float = float('inf')):
        self.decoder.train()
        
        print(f"Training with L-inf constraint: {epsilon:.4f}")
        print(f"Embedding bank size: {len(self.embedding_bank)}")
        print(f"Starting from epoch: {start_epoch + 1}")
        
        for epoch in range(start_epoch, num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                loss = self.train_step(batch, epsilon)
                total_loss += loss
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.4f}")
            
            avg_epoch_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.4f}")
            
            if (epoch + 1) % save_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'epsilon': epsilon,
                }
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint, checkpoint_path)
            
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(checkpoint, best_model_path)
                print(f"Best model saved with loss: {best_loss:.4f}")

    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * NOISE_IMAGE_SIZE * NOISE_IMAGE_SIZE * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def _assign_key_concepts(self, reasoning_chain, target_aspect_ratio, orig_width, orig_height):
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        key_concepts = [[] for _ in range(blocks)]
        
        target_width = NOISE_IMAGE_SIZE * target_aspect_ratio[0]
        target_height = NOISE_IMAGE_SIZE * target_aspect_ratio[1]
        
        for i in range(blocks):
            block_x = (i % target_aspect_ratio[0]) * NOISE_IMAGE_SIZE
            block_y = (i // target_aspect_ratio[0]) * NOISE_IMAGE_SIZE
            
            orig_block_x1 = block_x * orig_width / target_width
            orig_block_y1 = block_y * orig_height / target_height
            orig_block_x2 = (block_x + NOISE_IMAGE_SIZE) * orig_width / target_width
            orig_block_y2 = (block_y + NOISE_IMAGE_SIZE) * orig_height / target_height
            
            has_concept = False
            for step in reasoning_chain:
                if 'square_bbox' not in step:
                    continue
                
                bbox = step['square_bbox']
                
                if not isinstance(bbox, list) or len(bbox) != 3:
                    continue
                
                try:
                    center_x_norm, center_y_norm, size_norm = bbox
                    center_x = float(center_x_norm) * orig_width
                    center_y = float(center_y_norm) * orig_height
                    size = float(size_norm) * min(orig_width, orig_height)
                except (ValueError, TypeError, IndexError):
                    continue
                
                bbox_x1 = center_x - size / 2
                bbox_y1 = center_y - size / 2
                bbox_x2 = center_x + size / 2
                bbox_y2 = center_y + size / 2
                
                if (orig_block_x1 < bbox_x2 and orig_block_x2 > bbox_x1 and 
                    orig_block_y1 < bbox_y2 and orig_block_y2 > bbox_y1):
                    key_concepts[i].append(step['key_concept'])
                    has_concept = True
            
            if not has_concept:
                key_concepts[i] = [step['key_concept'] for step in reasoning_chain]
        
        return key_concepts
    
    def _dynamic_preprocess_with_info(self, image):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set(
            (i, j) for n in range(1, 12 + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= 12 and i * j >= 1
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height
        )
        
        target_width = NOISE_IMAGE_SIZE * target_aspect_ratio[0]
        target_height = NOISE_IMAGE_SIZE * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        resized_img = image.resize((target_width, target_height))
        
        processed_images = []
        for i in range(blocks):
            box = (
                (i % target_aspect_ratio[0]) * NOISE_IMAGE_SIZE,
                (i // target_aspect_ratio[0]) * NOISE_IMAGE_SIZE,
                ((i % target_aspect_ratio[0]) + 1) * NOISE_IMAGE_SIZE,
                ((i // target_aspect_ratio[0]) + 1) * NOISE_IMAGE_SIZE
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        return processed_images, target_aspect_ratio

    def generate_adversarial_image(self, image_path: str, reasoning_chain: list, 
                                 epsilon: float = 8.0/255.0, save_path: str = None):
        self.decoder.eval()
        
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        processed_images, target_aspect_ratio = self._dynamic_preprocess_with_info(image)
        
        transform = T.Compose([
            T.Resize((NOISE_IMAGE_SIZE, NOISE_IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        pixel_values = torch.stack([transform(img) for img in processed_images]).to(self.device)
        
        key_concepts = self._assign_key_concepts(
            reasoning_chain, target_aspect_ratio, orig_width, orig_height
        )
        
        with torch.no_grad():
            target_embeddings = self.get_target_embeddings(key_concepts)
            
            raw_noise_blocks = self.decoder(target_embeddings)
            noise_blocks = torch.clamp(raw_noise_blocks, -epsilon, epsilon)
            
            reconstructed_noise = self.tensor_processor.reconstruct_image(
                noise_blocks, target_aspect_ratio, (orig_width, orig_height)
            )
            
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
            
            if save_path:
                adversarial_image.save(save_path)
                print(f"Adversarial image saved to {save_path}")
            
            return adversarial_image

    def evaluate_adversarial_effectiveness(self, image_path: str, reasoning_chain: list, 
                                         epsilon: float = 8.0/255.0) -> dict:
        self.decoder.eval()
        
        adversarial_image = self.generate_adversarial_image(
            image_path, reasoning_chain, epsilon, save_path=None
        )
        
        original_image = Image.open(image_path).convert('RGB')
        
        transform = T.Compose([
            T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
        
        orig_tensor = transform(original_image).unsqueeze(0).to(self.device)
        adv_tensor = transform(adversarial_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            orig_features = self.clip_encoder.encode_img(orig_tensor)
            adv_features = self.clip_encoder.encode_img(adv_tensor)
            
            orig_features = F.normalize(orig_features, dim=-1)
            adv_features = F.normalize(adv_features, dim=-1)
            
            feature_similarity = F.cosine_similarity(orig_features, adv_features, dim=-1).item()
            
            concept_similarities = {}
            for step in reasoning_chain:
                concept = step['key_concept']
                
                text_embedding = self.clip_encoder.encode_text([concept], self.device)
                text_embedding = F.normalize(text_embedding, dim=-1)
                
                orig_concept_sim = F.cosine_similarity(orig_features, text_embedding, dim=-1).item()
                adv_concept_sim = F.cosine_similarity(adv_features, text_embedding, dim=-1).item()
                
                concept_similarities[concept] = {
                    'original': orig_concept_sim,
                    'adversarial': adv_concept_sim,
                    'reduction': orig_concept_sim - adv_concept_sim
                }
            
            noise_magnitude = torch.max(torch.abs(adv_tensor - orig_tensor)).item()
        
        return {
            'feature_similarity': feature_similarity,
            'concept_similarities': concept_similarities,
            'noise_magnitude': noise_magnitude,
            'epsilon_constraint': epsilon,
            'constraint_satisfied': noise_magnitude <= epsilon + 1e-6
        }

def main(args):
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    global NOISE_IMAGE_SIZE
    NOISE_IMAGE_SIZE = args.noise_image_size
    
    if args.verbose:
        print(f"Arguments: {args}")
        print(f"Using device: {device}")
        print(f"Noise image size: {NOISE_IMAGE_SIZE}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    dataset = PrivacyProtectionDataset(
        args.jsonl_path,
        image_root=args.image_root,
        image_size=args.noise_image_size,
        max_num=args.max_num_blocks,
        debug=args.debug
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        collate_fn=custom_collate_fn
    )
    
    trainer = AdversarialPrivacyTrainer(
        args.embedding_bank_path, 
        device, 
        lr=args.lr,
        debug=args.debug,
        surrogate_models=args.surrogate_models,
        use_ensemble=args.use_ensemble,
        use_fp16=args.use_fp16
    )
    
    start_epoch = 0
    best_loss = float('inf')
    epsilon = args.epsilon
    
    checkpoint_path = args.resume_from_checkpoint or args.pretrained_decoder
    if checkpoint_path:
        start_epoch, best_loss, loaded_epsilon = trainer.load_checkpoint(
            checkpoint_path, resume_training=bool(args.resume_from_checkpoint)
        )
        if loaded_epsilon is not None:
            epsilon = loaded_epsilon
    
    if not os.path.exists(args.embedding_bank_path):
        print("Building embedding bank...")
        trainer.embedding_bank = trainer.build_embedding_bank(dataset, args.embedding_bank_path)
    
    trainer.train(
        dataloader, 
        num_epochs=args.num_epochs, 
        epsilon=epsilon,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        verbose=args.verbose,
        start_epoch=start_epoch,
        best_loss=best_loss
    )
    
    if len(dataset) > 0:
        test_item = dataset.data[0]
        test_reasoning = test_item['location_analysis']['reasoning_chain']
        
        adversarial_img = trainer.generate_adversarial_image(
            test_item['image_path'],
            test_reasoning,
            epsilon=args.epsilon,
            save_path='test_adversarial.jpg'
        )
        
        evaluation_results = trainer.evaluate_adversarial_effectiveness(
            test_item['image_path'],
            test_reasoning,
            epsilon=args.epsilon
        )
        
        print("\n=== Adversarial Evaluation Results ===")
        print(f"Feature similarity: {evaluation_results['feature_similarity']:.4f}")
        print(f"Noise magnitude: {evaluation_results['noise_magnitude']:.6f}")
        
        for concept, sims in evaluation_results['concept_similarities'].items():
            print(f"  {concept}:")
            print(f"    Original: {sims['original']:.4f}")
            print(f"    Adversarial: {sims['adversarial']:.4f}")
            print(f"    Reduction: {sims['reduction']:.4f}")

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description='Adversarial Privacy Protection Training')
        
        parser.add_argument('--jsonl_path', type=str, default='data/json/location_analysis_fixed.jsonl',
                        help='Path to the JSONL dataset file')
        parser.add_argument('--image_root', type=str, default='',
                        help='Root directory for images (to join with relative paths in JSONL)')
        parser.add_argument('--embedding_bank_path', type=str, default='data/embedding_bank.pth',
                        help='Path to save/load embedding bank')
        parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
        parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
        parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
        parser.add_argument('--epsilon', type=float,
                        help='L-infinity constraint for adversarial noise')
        parser.add_argument('--noise_image_size', type=int, default=NOISE_IMAGE_SIZE,
                        help='Size of noise image blocks')
        parser.add_argument('--max_num_blocks', type=int,
                        help='Maximum number of blocks for dynamic preprocessing')
        parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
        parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for data loading')
        parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for image processing')
        parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
        parser.add_argument('--use_fp16', action='store_true',
                        default=True)
        parser.add_argument('--save_dir', type=str, default='./checkpoints/',
                        help='Directory to save checkpoints')
        parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
        parser.add_argument('--pretrained_decoder', type=str, default=None,
                           help='Path to pretrained decoder checkpoint (for initial training)')
        parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                           help='Path to checkpoint for resuming training (includes optimizer state)')
        parser.add_argument('--surrogate_models', type=str, nargs='+', 
                        default=[
                            'openai/clip-vit-base-patch32',
                            'openai/clip-vit-base-patch16',
                            'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                            'laion/CLIP-ViT-L-14-laion2B-s32B-b82K'
                        ],
                        help='List of surrogate models')
        parser.add_argument('--use_ensemble', action='store_true', default=True,
                        help='Use ensemble of surrogate models')
        
        return parser.parse_args()

    args = parse_args()
    main(args)