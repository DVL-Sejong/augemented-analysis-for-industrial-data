class Tokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos
        self.bos_idx = self.stoi["<bos>"]
        self.eos_idx = self.stoi["<eos>"]
        self.pad_idx = self.stoi["<pad>"]
        self.unk_idx = self.stoi["<unk>"]

    def encode(self, text):
        tokens = text.lower().split()
        token_ids = [self.bos_idx]
        for t in tokens:
            token_ids.append(self.stoi.get(t, self.unk_idx))
        token_ids.append(self.eos_idx)
        return token_ids

    def decode(self, token_ids):
        words = []
        for t in token_ids:
            if t in [self.bos_idx, self.eos_idx, self.pad_idx]:
                continue
            words.append(self.itos.get(t, "<unk>"))
        return " ".join(words)
