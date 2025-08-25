import torch
import argparse
from torchvision import transforms
from model import VideoCaptionModel
from tokenizer import Tokenizer
import torchvision.io as io

def uniform_sample_indices(total, num):
    return [int(i * total / num) for i in range(num)]

def load_video_frames(video_path, max_frames=16):
    vr, _, _ = io.read_video(video_path, pts_unit='sec')
    total_frames = vr.shape[0]
    if total_frames == 0:
        raise ValueError(f"No frames found in {video_path}")
    if total_frames < max_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (max_frames - total_frames)
    else:
        indices = uniform_sample_indices(total_frames, max_frames)
    selected_frames = vr[indices]
    selected_frames = selected_frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    return selected_frames

def main(video_path, max_len):
    checkpoint = torch.load("video_caption_model.pth", map_location="cpu")
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    tokenizer = Tokenizer(stoi, itos)

    transform = transforms.Compose([
        transforms.ConvertImageDtype(torch.float),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VideoCaptionModel(vocab_size=len(stoi), embed_dim=512, pad_idx=tokenizer.pad_idx).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    frames = load_video_frames(video_path, max_frames=16)
    frames = torch.stack([transform(f) for f in frames])  # (T, C, H, W)
    frames = frames.unsqueeze(0).to(device)  # (B=1, T, C, H, W)

    with torch.no_grad():
        enc_feats = model.video_encoder(frames)
        ys = model.decoder.generate(enc_feats, tokenizer.bos_idx, tokenizer.eos_idx, max_len=max_len)

    caption = tokenizer.decode(ys[0].cpu().numpy())
    print("Generated Caption:", caption)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Captioning Inference")
    parser.add_argument("--video_path", type=str, help="Path to the input video file")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum length of the generated caption")
    
    args = parser.parse_args()
    main(args.video_path, args.max_len)