import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import models

class VideoEncoder(nn.Module):
    def __init__(self, feat_dim=2048, embed_dim=512, num_layers=4, nheads=8):
        super().__init__()
        base_model = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        self.fc = nn.Linear(feat_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nheads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, frames):
        B, T, C, H, W = frames.size()
        feat_list = []
        with torch.no_grad():
            for t in range(T):
                f = self.feature_extractor(frames[:, t])
                f = f.view(B, 2048)
                f = self.fc(f)
                feat_list.append(f)
        feats = torch.stack(feat_list, dim=1)
        feats = rearrange(feats, 'b t d -> t b d')
        feats = self.transformer_encoder(feats)
        return rearrange(feats, 't b d -> b t d')

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=4, nheads=8, max_len=50, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nheads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def forward(self, enc_feats, tgt_tokens):
        enc_feats = rearrange(enc_feats, 'b t d -> t b d')
        tgt_emb = self.embedding(tgt_tokens) # (B, L, D)
        tgt_emb = rearrange(tgt_emb, 'b l d -> l b d')
        
        tgt_mask = self._generate_square_subsequent_mask(tgt_emb.size(0)).to(tgt_emb.device)
        output = self.transformer_decoder(tgt_emb, enc_feats, tgt_mask=tgt_mask)
        output = rearrange(output, 'l b d -> b l d')
        logits = self.fc_out(output)
        return logits
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask

    def generate(self, enc_feats, start_token, end_token, max_len=50):
        enc_feats = rearrange(enc_feats, 'b t d -> t b d')
        B = enc_feats.size(1)
        ys = torch.full((B,1), start_token, dtype=torch.long, device=enc_feats.device)
        for i in range(max_len-1):
            tgt_mask = self._generate_square_subsequent_mask(ys.size(1)).to(enc_feats.device)
            tgt_emb = rearrange(self.embedding(ys), 'b l d -> l b d')
            out = self.transformer_decoder(tgt_emb, enc_feats, tgt_mask=tgt_mask)
            out = rearrange(out, 'l b d -> b l d')
            prob = self.fc_out(out[:, -1, :]) # (B, vocab)
            next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
            ys = torch.cat([ys, next_word], dim=1)
            if (next_word == end_token).all():
                break
        return ys

class VideoCaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, pad_idx=0):
        super().__init__()
        self.video_encoder = VideoEncoder(embed_dim=embed_dim)
        self.decoder = CaptionDecoder(vocab_size=vocab_size, embed_dim=embed_dim, pad_idx=pad_idx)

    def forward(self, frames, tgt_tokens):
        enc_feats = self.video_encoder(frames)
        logits = self.decoder(enc_feats, tgt_tokens)
        return logits
