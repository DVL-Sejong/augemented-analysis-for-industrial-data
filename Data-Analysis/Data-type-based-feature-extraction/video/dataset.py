import torch
from torch.utils.data import Dataset
import json
import random
import os
from torchvision import transforms
import torchvision.io as io

class MSRVTTDataset(Dataset):
    def __init__(self, annotation_file, video_root, tokenizer, max_frames=16, transform=None, split='train', max_videos=None):
        self.annotation = json.load(open(annotation_file,'r'))
        self.video_root = video_root
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.transform = transform
        self.split = split

        videos = self.annotation["videos"]
        sentences = self.annotation["sentences"]

        self.video_entries = [v for v in videos if v["split"] == self.split]

        self.video_captions = {}
        for s in sentences:
            vid = s["video_id"]
            caption = s["caption"]
            if vid not in self.video_captions:
                self.video_captions[vid] = []
            self.video_captions[vid].append(caption)

        self.video_entries = [v for v in self.video_entries if v["video_id"] in self.video_captions]

        if max_videos is not None:
            self.video_entries = self.video_entries[:max_videos]

    def __len__(self):
        return len(self.video_entries)

    def __getitem__(self, idx):
        entry = self.video_entries[idx]
        video_id = entry["video_id"]
        captions = self.video_captions[video_id]
        caption = random.choice(captions)
        frames = self._load_video_frames(video_id, self.max_frames)

        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        else:
            frames = frames

        tokens = self.tokenizer.encode(caption)
        return frames, torch.tensor(tokens, dtype=torch.long)

    def _load_video_frames(self, video_id, max_frames):
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")
        vr,_,_ = io.read_video(video_path, pts_unit='sec')
        total_frames = vr.shape[0]
        if total_frames < max_frames:
            indices = list(range(total_frames)) + [total_frames-1]*(max_frames - total_frames)
        else:
            indices = self._uniform_sample_indices(total_frames, max_frames)
        selected_frames = vr[indices]
        selected_frames = selected_frames.permute(0,3,1,2)
        return selected_frames

    def _uniform_sample_indices(self, total, num):
        return [int(i*total/num) for i in range(num)]
