import cv2, argparse, tempfile, shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from packaging import version
from transformers import pipeline
from huggingface_hub import login



if version.parse(torch.__version__) < version.parse("2.5"):
    raise RuntimeError("torch 2.5 이상 필요")
print(f"[info] torch {torch.__version__}, transformers import OK")

def extract_keyframes(video_path: str,
                      fps_interval: float = 0.5,
                      out_dir: Path | None = None) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Can't open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hop = max(1, int(round(fps * fps_interval)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = out_dir or Path(tempfile.mkdtemp())
    frames = []

    for i in tqdm(range(0, total, hop), desc="extract", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        p = out_dir / f"f_{i:06d}.jpg"
        cv2.imwrite(str(p), frame)
        frames.append(p)

    cap.release()
    return frames

CAPTION_MODEL = "microsoft/git-large-coco"
def caption_frames(frame_paths: list[Path], batch_size: int = 8) -> list[str]:
    device = 0 if torch.cuda.is_available() else -1
    capt_pipe = pipeline(
        "image-to-text",
        model=CAPTION_MODEL,
        device=device,
        batch_size=batch_size,
        trust_remote_code=True,
        model_kwargs={"use_safetensors": True},
        generate_kwargs={"max_new_tokens": 32},
    )

    def _to_text(o):
        if isinstance(o, dict) and "generated_text" in o:
            return o["generated_text"].strip()
        elif isinstance(o, str):
            return o.strip()
        else:
            return str(o).strip()

    captions = []
    for i in tqdm(range(0, len(frame_paths), batch_size),
                  desc="caption", unit="frame"):
        batch_imgs = [Image.open(p).convert("RGB") for p in frame_paths[i:i+batch_size]]
        outs = capt_pipe(batch_imgs)
        captions.extend(_to_text(x) for x in outs)

    return captions

SUMMARIZER_MODEL = "facebook/bart-large-cnn"
def summarize_captions(captions: list[str]) -> str:
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline(
        "summarization",
        model=SUMMARIZER_MODEL,
        device=device,
        trust_remote_code=True,
        model_kwargs={"use_safetensors": True},
        min_length=15, max_length=120,
    )
    joined = " ".join(captions)
    out = summarizer(joined, truncation=True)
    return out[0]["summary_text"].strip()

POST_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
def polish_text(raw_summary):
    prompt = (
        "Rewrite the following description as one coherent, natural paragraph. "
        "Keep the meaning, remove repetitions, and use smooth transitions.\n\n"
        f"{raw_summary}\n\nPolished paragraph:"
    )
    gen = pipeline(
        "text-generation",
        model=POST_MODEL,
        device=0 if torch.cuda.is_available() else -1,
        model_kwargs={"use_safetensors": True},
        max_new_tokens=120,
        temperature=0.7,
    )
    return gen(prompt)[0]["generated_text"].split("Polished paragraph:")[-1].strip()

def video_to_text(video_path: str,
                  fps_interval: float = 0.5) -> str:
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        frames = extract_keyframes(video_path, fps_interval, tmp_dir)
        captions = caption_frames(frames)
        summary = summarize_captions(captions)
        return summary
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="video → descriptive text")
    ap.add_argument("video", help="input video file")
    ap.add_argument("--fps_interval", type=float, default=0.5,
                    help="seconds between sampled frames (default 0.5)")
    args = ap.parse_args()

    text = video_to_text(args.video, fps_interval=args.fps_interval)
    print("\n=== FINAL SUMMARY ===\n")
    print(text)
    print("\n=== POLISHING SUMMARY ===\n")
    print(polish_text(text))
