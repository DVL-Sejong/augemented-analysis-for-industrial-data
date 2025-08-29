import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import MSRVTTDataset
from model import VideoCaptionModel
from build_vocab import build_vocab
from tokenizer import Tokenizer

def collate_fn(batch):
    frames_batch = []
    tokens_batch = []
    max_len = max([x[1].size(0) for x in batch])
    for frames, tokens in batch:
        frames_batch.append(frames)
        padded = torch.full((max_len,), fill_value=tokenizer.pad_idx, dtype=torch.long)
        length = tokens.size(0)
        padded[:length] = tokens
        tokens_batch.append(padded)
    frames_batch = torch.stack(frames_batch)
    tokens_batch = torch.stack(tokens_batch)
    return frames_batch, tokens_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_false', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, default='video_caption_model.pth', help='Path to checkpoint file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    args = parser.parse_args()

    annotation_file = "train_val_videodatainfo.json"
    video_root = "videos/"
    
    stoi, itos = build_vocab(annotation_file, max_words=10000, min_freq=1)
    tokenizer = Tokenizer(stoi, itos)

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    dataset = MSRVTTDataset(annotation_file, video_root, tokenizer, max_frames=16, transform=transform, split='train', max_videos=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = VideoCaptionModel(vocab_size=len(stoi), embed_dim=512, pad_idx=tokenizer.pad_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        stoi = checkpoint["stoi"]
        itos = checkpoint["itos"]
        tokenizer = Tokenizer(stoi, itos)

    epochs = args.epochs
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        epoch_loss_sum = 0.0
        batch_count = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{start_epoch+epochs}]", ncols=100)
        for i, (frames, tokens) in enumerate(progress_bar):
            frames = frames.to(device)
            tokens = tokens.to(device)
            input_tokens = tokens[:, :-1]
            target_tokens = tokens[:, 1:]

            logits = model(frames, input_tokens)
            loss = loss_fn(logits.transpose(1,2), target_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss_sum += batch_loss
            batch_count += 1

            progress_bar.set_postfix(loss=f"{batch_loss:.4f}")

        avg_loss = epoch_loss_sum / batch_count
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_loss:.4f}")

        torch.save({
            "model": model.state_dict(),
            "stoi": stoi,
            "itos": itos
        }, args.checkpoint)
