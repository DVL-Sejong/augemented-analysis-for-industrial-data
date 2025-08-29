import json
from collections import Counter

def tokenize_caption(caption):
    caption = caption.lower()
    tokens = caption.split()
    return tokens

def build_vocab(annotation_file, max_words=10000, min_freq=1, special_tokens=["<bos>", "<eos>", "<pad>", "<unk>"]):
    annotation = json.load(open(annotation_file, 'r'))
    videos = annotation["videos"]
    sentences = annotation["sentences"]

    train_video_ids = set([v["video_id"] for v in videos if v["split"] == "train"])
    train_captions = [s["caption"] for s in sentences if s["video_id"] in train_video_ids]

    counter = Counter()
    for cap in train_captions:
        tokens = tokenize_caption(cap)
        counter.update(tokens)

    words = [w for w, c in counter.items() if c >= min_freq]
    words = sorted(words, key=lambda x: counter[x], reverse=True)
    words = words[:max_words]

    vocab_words = special_tokens + words
    stoi = {w:i for i,w in enumerate(vocab_words)}
    itos = {i:w for i,w in enumerate(vocab_words)}

    return stoi, itos
