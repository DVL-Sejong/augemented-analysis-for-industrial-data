# random_yt_transcript_crawl.py
# ─────────────────────────────
"""
1. YouTube Data API v3로 랜덤 영상 100개 수집
2. YouTubeTranscriptAPI ▸ 실패 ▶ yt-dlp 자막 ▸ 실패 ▶ Whisper STT
3. 성공/실패 결과, 실패 사유 CSV 두 개로 저장
"""

# === 기본 모듈 ===========================================================
import os, random, time, csv, logging, itertools, re, tempfile, sys
from xml.etree.ElementTree import ParseError

# === 외부 라이브러리 ====================================================
from googleapiclient.discovery import build
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
import webvtt                                   # pip install webvtt-py

# === 사용자 설정 ========================================================
API_KEY        = "AIzaSyC14u6gwZSQxdpkTkPkJ79k4_hyBtwReGM"
NUM_VIDEOS     = 50
TARGET_LANG    = "en"           # ko / en / etc.
USE_WHISPER    = True           # False면 Whisper 건너뜀
WHISPER_SIZE   = "tiny"         # tiny | base | small ...
COOKIE_FILE    = "cookies.txt"  # 로그인 필요 영상이면 브라우저 쿠키 추출
FFMPEG_BIN     = "ffmpeg"       # 시스템 PATH면 그냥 "ffmpeg"

# === 유틸: 랜덤 영상 뽑기 ==============================================
def youtube_service():
    return build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)

def random_query():
    letters = "abcdefghijklmnopqrstuvwxyz"
    return "".join(random.choice(letters) for _ in range(2))

def fetch_random_video_ids(n=NUM_VIDEOS):
    svc, ids = youtube_service(), set()
    while len(ids) < n:
        q = random_query()
        res = svc.search().list(q=q, part="id", maxResults=50, type="video").execute()
        ids.update(item["id"]["videoId"] for item in res["items"])
    return list(ids)[:n]

# === 단계 1: YouTubeTranscriptAPI ======================================
def try_yta(video_id: str):
    try:
        data = YouTubeTranscriptApi.get_transcript(
            video_id, languages=[TARGET_LANG, f"{TARGET_LANG}-fr", f"{TARGET_LANG}-en"]
        )
        return " ".join(piece["text"] for piece in data), "YTA", ""
    except (TranscriptsDisabled, NoTranscriptFound, ParseError) as e:
        return None, "", f"yta:{type(e).__name__}"
    except Exception as e:
        return None, "", f"yta:{type(e).__name__}"

# === 단계 2: yt-dlp + VTT ==============================================
def try_ytdlp(video_id: str):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            outtmpl = os.path.join(tmp, "%(id)s.%(ext)s")
            opts = {
                "skip_download": True,
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": [
                    TARGET_LANG,
                    f"{TARGET_LANG}-fr",
                    f"{TARGET_LANG}-en",
                ],
                "outtmpl": outtmpl,
                "quiet": True,
                "no_warnings": True,
                "geo_bypass": True,
                "geo_bypass_country": "US",
                "cookiefile": COOKIE_FILE if os.path.exists(COOKIE_FILE) else None,
                "ffmpeg_location": FFMPEG_BIN,
            }
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_id, download=False)
                if not (info.get("subtitles") or info.get("automatic_captions")):
                    return None, "", "ytdlp:no_sub"
                ydl.download([video_id])

            vtts = [f for f in os.listdir(tmp) if f.endswith(".vtt")]
            if not vtts:
                return None, "", "ytdlp:no_vtt"

            path = os.path.join(tmp, vtts[0])
            chunks = []
            for cue in webvtt.read(path):
                txt = re.sub(r"<[^>]+>", "", cue.text).replace("\n", " ").strip()
                if txt:
                    chunks.append(txt)
            chunks = [k for k, _ in itertools.groupby(chunks)]
            return " ".join(chunks), "YDL", ""
    except DownloadError as e:
        return None, "", f"ytdlp:DownloadError:{e.exc_info[0]}"
    except Exception as e:
        return None, "", f"ytdlp:{type(e).__name__}"

# === 단계 3: Whisper ====================================================
def try_whisper(video_id: str):
    if not USE_WHISPER:
        return None, "", "skip_whisper"
    try:
        import whisper, torch
        audio_tmp = tempfile.mktemp(suffix=".m4a")
        dl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_tmp,
            "quiet": True,
            "no_warnings": True,
            "cookiefile": COOKIE_FILE if os.path.exists(COOKIE_FILE) else None,
            "geo_bypass": True,
            "geo_bypass_country": "US",
            "ffmpeg_location": FFMPEG_BIN,
        }
        with YoutubeDL(dl_opts) as ydl:
            ydl.download([f"https://youtu.be/{video_id}"])

        if not (os.path.exists(audio_tmp) and os.path.getsize(audio_tmp) > 0):
            return None, "", "whisper:no_audio"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(WHISPER_SIZE, device=device)
        res = model.transcribe(audio_tmp, language=None, fp16=False)
        os.remove(audio_tmp)
        return res["text"], "WSP", ""
    except DownloadError as e:
        return None, "", f"whisper:DownloadError:{e.exc_info[0]}"
    except Exception as e:
        return None, "", f"whisper:{type(e).__name__}:{e}"

# === 파이프라인 통합 ====================================================
def get_transcript(video_id: str):
    for fn in (try_yta, try_ytdlp, try_whisper):
        txt, tag, reason = fn(video_id)
        if txt:
            return txt, tag, ""
        if tag or reason:
            last_reason = reason
    return None, "FAIL", last_reason

# === 메인 루프 ==========================================================
def main():
    if API_KEY == "YOUR_REAL_API_KEY":
        sys.exit("❌  API_KEY 설정 안 했음")
    random.seed(time.time())
    #vids = fetch_random_video_ids()
    #print(f"▶ 영상 {len(vids)}개 랜덤 수집 완료")

    vids = ['mR7G5HKx814', 'pSLg791dTiQ']

    stats = {"YTA": 0, "YDL": 0, "WSP": 0, "FAIL": 0}
    with open("transcript_ok.csv", "w", newline="", encoding="utf-8") as ok, \
         open("transcript_fail.csv", "w", newline="", encoding="utf-8") as ng:
        ok_w, ng_w = csv.writer(ok), csv.writer(ng)
        ok_w.writerow(["video_id", "method", "length"])
        ng_w.writerow(["video_id", "reason"])

        for i, vid in enumerate(vids, 1):
            txt, method, reason = get_transcript(vid)
            stats[method] += 1
            if method != "FAIL":
                ok_w.writerow([vid, method, len(txt)])
            else:
                ng_w.writerow([vid, reason])
            print(f"[{i:3}/{len(vids)}] {vid} → {method} ({reason})")

    print("\n=== 요약 ===")
    for k, v in stats.items():
        print(f"{k:4}: {v}")
    print("성공 파일: transcript_ok.csv")
    print("실패 파일: transcript_fail.csv")

if __name__ == "__main__":
    logging.getLogger("youtube_transcript_api").setLevel(logging.CRITICAL)
    main()
