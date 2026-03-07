from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
try:
    from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
except ImportError:
    TranscriptsDisabled = type('TranscriptsDisabled', (Exception,), {})
    NoTranscriptFound = type('NoTranscriptFound', (Exception,), {})
    VideoUnavailable = type('VideoUnavailable', (Exception,), {})
from pydantic import BaseModel
from typing import Optional
import re, os, json, httpx, time
import yt_dlp

app = FastAPI(title="UCreatorKit API v3", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET","POST"], allow_headers=["*"])

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL  = "claude-sonnet-4-20250514"

# 30 supported languages
LANGUAGES = {
    "en":    {"name": "English",            "flag": "🇺🇸"},
    "ko":    {"name": "Korean",             "flag": "🇰🇷"},
    "ja":    {"name": "Japanese",           "flag": "🇯🇵"},
    "zh-Hans":{"name":"Chinese (Simplified)","flag": "🇨🇳"},
    "zh-Hant":{"name":"Chinese (Traditional)","flag":"🇹🇼"},
    "es":    {"name": "Spanish",            "flag": "🇪🇸"},
    "fr":    {"name": "French",             "flag": "🇫🇷"},
    "de":    {"name": "German",             "flag": "🇩🇪"},
    "pt":    {"name": "Portuguese",         "flag": "🇧🇷"},
    "pt-PT": {"name": "Portuguese (EU)",    "flag": "🇵🇹"},
    "ru":    {"name": "Russian",            "flag": "🇷🇺"},
    "ar":    {"name": "Arabic",             "flag": "🇸🇦"},
    "hi":    {"name": "Hindi",              "flag": "🇮🇳"},
    "it":    {"name": "Italian",            "flag": "🇮🇹"},
    "nl":    {"name": "Dutch",              "flag": "🇳🇱"},
    "pl":    {"name": "Polish",             "flag": "🇵🇱"},
    "tr":    {"name": "Turkish",            "flag": "🇹🇷"},
    "id":    {"name": "Indonesian",         "flag": "🇮🇩"},
    "ms":    {"name": "Malay",              "flag": "🇲🇾"},
    "th":    {"name": "Thai",               "flag": "🇹🇭"},
    "vi":    {"name": "Vietnamese",         "flag": "🇻🇳"},
    "sv":    {"name": "Swedish",            "flag": "🇸🇪"},
    "da":    {"name": "Danish",             "flag": "🇩🇰"},
    "fi":    {"name": "Finnish",            "flag": "🇫🇮"},
    "no":    {"name": "Norwegian",          "flag": "🇳🇴"},
    "cs":    {"name": "Czech",              "flag": "🇨🇿"},
    "ro":    {"name": "Romanian",           "flag": "🇷🇴"},
    "hu":    {"name": "Hungarian",          "flag": "🇭🇺"},
    "uk":    {"name": "Ukrainian",          "flag": "🇺🇦"},
    "fa":    {"name": "Persian (Farsi)",    "flag": "🇮🇷"},
}

async def call_claude(system: str, user: str, max_tokens: int = 1000) -> str:
    if not ANTHROPIC_KEY:
        raise HTTPException(503, "Set ANTHROPIC_API_KEY environment variable to use AI features.")
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": CLAUDE_MODEL, "max_tokens": max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}]},
        )
        if res.status_code != 200:
            raise HTTPException(502, f"AI API error: {res.text}")
        return res.json()["content"][0]["text"]

def extract_video_id(url: str) -> str:
    for p in [r"(?:v=)([A-Za-z0-9_-]{11})",r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
              r"(?:embed/)([A-Za-z0-9_-]{11})",r"(?:shorts/)([A-Za-z0-9_-]{11})"]:
        m = re.search(p, url)
        if m: return m.group(1)
    raise HTTPException(400, "Invalid YouTube URL.")

def fmt_ts(s: float) -> str:
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{int(s%60):02d}"

def _ts_srt(s: float) -> str:
    h, rem = divmod(s, 3600); m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{int((sec%1)*1000):03d}"

def _ts_vtt(s: float) -> str:
    h, rem = divmod(s, 3600); m, sec = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(sec):02d}.{int((sec%1)*1000):03d}"

def format_srt(entries):
    lines = []
    for i, e in enumerate(entries, 1):
        s, d = e["start"], e["duration"]
        lines.append(f"{i}\n{_ts_srt(s)} --> {_ts_srt(s+d)}\n{e['text']}")
    return "\n\n".join(lines)

def format_txt(entries):
    return "\n".join(e["text"] for e in entries)

def format_vtt(entries):
    lines = ["WEBVTT\n"]
    for e in entries:
        s, d = e["start"], e["duration"]
        lines.append(f"{_ts_vtt(s)} --> {_ts_vtt(s+d)}\n{e['text']}")
    return "\n\n".join(lines)

def get_transcript_ytdlp(video_id: str, lang: str):
    """Fallback transcript extraction using yt-dlp"""
    import tempfile, os
    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': [lang, 'en'],
            'subtitlesformat': 'json3',
            'skip_download': True,
            'outtmpl': os.path.join(tmpdir, '%(id)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'format': 'worst',
            'ignore_no_formats_error': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception:
            # Retry without format option
            ydl_opts.pop('format', None)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        # Find subtitle file
        for l in [lang, 'en']:
            fpath = os.path.join(tmpdir, f"{video_id}.{l}.json3")
            if os.path.exists(fpath):
                with open(fpath) as f:
                    raw = json.load(f)
                entries = []
                for ev in raw.get('events', []):
                    segs = ev.get('segs', [])
                    text = ''.join(s.get('utf8','') for s in segs).strip()
                    if text and text != '\n':
                        start = ev.get('tStartMs', 0) / 1000
                        dur = ev.get('dDurationMs', 2000) / 1000
                        entries.append({'text': text, 'start': start, 'duration': dur})
                if entries:
                    lname = info.get('subtitles', {}).get(l, [{}])[0].get('name', l) if info else l
                    return entries, lname, l
    raise HTTPException(status_code=404, detail="No subtitles found for this video.")

def get_transcript_data(video_id: str, lang: str):
    last_err = None
    # Step 1: Try youtube-transcript-api with requested language
    for attempt in range(3):
        try:
            ytt_api = YouTubeTranscriptApi()
            tlist = ytt_api.list(video_id)
            t = None
            try: t = tlist.find_transcript([lang])
            except NoTranscriptFound:
                try:
                    for x in tlist:
                        if x.is_translatable: t = x.translate(lang); break
                except: pass
            if t is None: t = next(iter(tlist))
            fetched = t.fetch()
            data = fetched.to_raw_data()
            return data, t.language, t.language_code
        except (TranscriptsDisabled, VideoUnavailable):
            raise
        except Exception as e:
            last_err = e
            if "429" in str(e) and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            break
    # Step 2: If 429, try getting ANY available transcript (skip translation)
    if last_err and "429" in str(last_err):
        try:
            ytt_api = YouTubeTranscriptApi()
            tlist = ytt_api.list(video_id)
            t = next(iter(tlist))
            fetched = t.fetch()
            data = fetched.to_raw_data()
            return data, t.language, t.language_code
        except Exception:
            pass
    # Step 3: Fallback to yt-dlp
    for attempt in range(2):
        try:
            return get_transcript_ytdlp(video_id, lang)
        except HTTPException:
            raise
        except Exception as e2:
            last_err = e2
            if "429" in str(e2) and attempt < 1:
                time.sleep(2)
                continue
    # Step 4: Last resort - yt-dlp with English
    if lang != "en":
        try:
            return get_transcript_ytdlp(video_id, "en")
        except Exception:
            pass
    raise HTTPException(status_code=500, detail=f"Transcript extraction failed: {str(last_err)}")

def parse_json_safe(text: str) -> dict:
    clean = text.strip().replace("```json","").replace("```","").strip()
    return json.loads(clean)

# ── 01 Transcript ────────────────────────────────
@app.get("/api/transcript")
def get_transcript(url: str=Query(...), lang: str=Query("en"), format: str=Query("json")):
    vid = extract_video_id(url)
    try: data, lname, lcode = get_transcript_data(vid, lang)
    except TranscriptsDisabled: raise HTTPException(404, "Subtitles are disabled for this video.")
    except VideoUnavailable:    raise HTTPException(404, "Video not found.")
    except Exception as e:      raise HTTPException(500, f"Transcript error: {e}")
    if format == "srt": return {"format":"srt","content":format_srt(data)}
    if format == "txt": return {"format":"txt","content":format_txt(data)}
    if format == "vtt": return {"format":"vtt","content":format_vtt(data)}
    return {"format":"json","video_id":vid,"language":lname,"language_code":lcode,"entry_count":len(data),
            "entries":[{"start":round(e["start"],2),"duration":round(e["duration"],2),
                        "text":e["text"],"timestamp":fmt_ts(e["start"])} for e in data]}

@app.get("/api/transcript/languages")
def get_languages(url: str=Query(...)):
    vid = extract_video_id(url)
    try:
        ytt_api = YouTubeTranscriptApi()
        tlist = ytt_api.list(vid)
        langs = [{"code":t.language_code,"name":t.language,
                  "is_generated":t.is_generated,"is_translatable":t.is_translatable} for t in tlist]
        return {"video_id":vid,"total":len(langs),
                "manual":sum(1 for l in langs if not l["is_generated"]),
                "auto":sum(1 for l in langs if l["is_generated"]),"languages":langs}
    except TranscriptsDisabled: raise HTTPException(404, "Subtitles disabled.")
    except Exception as e: raise HTTPException(500, str(e))

# ── 02 AI Summary ────────────────────────────────
@app.get("/api/summary")
async def get_summary(url: str=Query(...), lang: str=Query("en")):
    vid = extract_video_id(url)
    try: data, _, _ = get_transcript_data(vid, lang)
    except Exception as e: raise HTTPException(500, f"Transcript error: {e}")
    full_text = " ".join(e["text"] for e in data)[:4000]
    result = await call_claude(
        "You are a YouTube video summarization expert. Analyze transcripts and extract key insights.",
        f"""Analyze this YouTube transcript and respond in JSON only:

Transcript: {full_text}

Output format:
{{"title_guess":"one-line topic summary","summary":"3-5 sentence summary","key_points":["point1","point2","point3"],"keywords":["kw1","kw2","kw3","kw4","kw5"]}}

Output JSON only, no other text.""", 800)
    try: return {"video_id": vid, **parse_json_safe(result)}
    except: raise HTTPException(500, "Failed to parse AI response.")

# ── 03 Thumbnail ─────────────────────────────────
@app.get("/api/thumbnail")
def get_thumbnail(url: str=Query(...)):
    vid = extract_video_id(url)
    base = f"https://img.youtube.com/vi/{vid}"
    return {"video_id":vid,"thumbnails":{"maxres":f"{base}/maxresdefault.jpg","hq":f"{base}/hqdefault.jpg",
            "mq":f"{base}/mqdefault.jpg","sd":f"{base}/sddefault.jpg"}}

# ── 04 Tags ──────────────────────────────────────
class TagRequest(BaseModel):
    topic: str
    lang: Optional[str] = "en"

@app.post("/api/tags/generate")
async def generate_tags(req: TagRequest):
    result = await call_claude(
        "You are a YouTube SEO expert. Generate optimized tags and keywords.",
        f"""Topic: "{req.topic}"

Output JSON only:
{{"tags":["tag1","tag2",...20+ tags],"short_tail":["keyword1","keyword2","keyword3"],"long_tail":["long tail phrase1","phrase2","phrase3"],"hashtags":["#hash1","#hash2","#hash3","#hash4","#hash5"]}}

Output JSON only.""", 1000)
    try: return parse_json_safe(result)
    except: raise HTTPException(500, "AI parse error.")

@app.get("/api/tags/extract")
def extract_tags(url: str=Query(...)):
    try:
        with yt_dlp.YoutubeDL({"quiet":True,"skip_download":True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return {"video_id":info.get("id"),"title":info.get("title"),
                "tags":(info.get("tags") or [])[:50],"categories":info.get("categories") or []}
    except Exception as e: raise HTTPException(500, f"Tag extraction failed: {e}")

# ── 05 Title Analysis ────────────────────────────
class TitleRequest(BaseModel):
    title: str
    topic: Optional[str] = ""

@app.post("/api/title/analyze")
async def analyze_title(req: TitleRequest):
    result = await call_claude(
        "You are a YouTube CTR optimization expert. Analyze titles and suggest improvements.",
        f"""Title: "{req.title}"{f' / Topic: {req.topic}' if req.topic else ''}

Output JSON only:
{{"ctr_score":75,"ctr_grade":"B+","strengths":["strength1","strength2"],"weaknesses":["weakness1","weakness2"],"suggestions":[{{"title":"improved title 1","reason":"reason"}},{{"title":"improved title 2","reason":"reason"}},{{"title":"improved title 3","reason":"reason"}}],"tips":["tip1","tip2","tip3"]}}

ctr_score is 0-100. Output JSON only.""", 1000)
    try: return parse_json_safe(result)
    except: raise HTTPException(500, "AI parse error.")

# ── 06 Channel Stats ─────────────────────────────
@app.get("/api/channel")
def get_channel(url: str=Query(...)):
    try:
        with yt_dlp.YoutubeDL({"quiet":True,"skip_download":True,"extract_flat":True,"playlistend":10}) as ydl:
            info = ydl.extract_info(url, download=False)
        entries = info.get("entries") or []
        views = [e.get("view_count",0) for e in entries if e.get("view_count")]
        return {
            "channel_id":       info.get("id"),
            "channel_name":     info.get("title") or info.get("uploader"),
            "channel_url":      info.get("webpage_url"),
            "description":      (info.get("description") or "")[:300],
            "subscriber_count": info.get("channel_follower_count"),
            "video_count":      info.get("playlist_count"),
            "thumbnail":        info.get("thumbnail"),
            "avg_views_recent10": int(sum(views)/len(views)) if views else 0,
            "recent_videos": [{"id":e.get("id"),"title":e.get("title"),"view_count":e.get("view_count"),
                "upload_date":e.get("upload_date"),"duration":e.get("duration"),
                "thumbnail":f"https://img.youtube.com/vi/{e.get('id')}/mqdefault.jpg" if e.get("id") else None}
                for e in entries[:10]],
        }
    except Exception as e: raise HTTPException(500, f"Channel analysis failed: {e}")

# ── 07 Translation (30 languages) ────────────────
@app.get("/api/languages")
def get_supported_languages():
    return {"languages": [{"code": k, "name": v["name"], "flag": v["flag"]} for k, v in LANGUAGES.items()]}

@app.get("/api/translate")
async def translate_transcript(
    url:      str = Query(...),
    src_lang: str = Query("en"),
    tgt_lang: str = Query("ko"),
):
    vid = extract_video_id(url)
    try: data, _, _ = get_transcript_data(vid, src_lang)
    except Exception as e: raise HTTPException(500, f"Transcript error: {e}")

    entries  = data[:60]
    batch    = "\n".join(f"{i}|{e['text']}" for i, e in enumerate(entries))
    tgt_info = LANGUAGES.get(tgt_lang, {"name": tgt_lang, "flag": ""})
    tgt_name = tgt_info["name"]

    result = await call_claude(
        f"You are a professional translator. Translate the given text accurately into {tgt_name}.",
        f"""Translate each subtitle line into {tgt_name}. Keep the 'number|text' format. Only translate the text part.

{batch}

Output 'number|translated_text' format only.""", 2000)

    translated = {}
    for line in result.strip().split("\n"):
        if "|" in line:
            idx, txt = line.split("|", 1)
            try: translated[int(idx.strip())] = txt.strip()
            except: pass

    return {
        "video_id": vid, "src_lang": src_lang, "tgt_lang": tgt_lang,
        "tgt_name": tgt_name, "tgt_flag": tgt_info.get("flag",""),
        "entries": [{"start":entries[i]["start"],"duration":entries[i]["duration"],
            "timestamp":fmt_ts(entries[i]["start"]),"original":entries[i]["text"],
            "translated":translated.get(i, entries[i]["text"])} for i in range(len(entries))],
    }

# ── 08 Chapter Generator ─────────────────────────
@app.get("/api/chapters")
async def generate_chapters(
    url:  str = Query(...),
    lang: str = Query("en"),
):
    vid = extract_video_id(url)

    # Fetch full transcript with timestamps
    try:
        data, _, _ = get_transcript_data(vid, lang)
    except TranscriptsDisabled: raise HTTPException(404, "Subtitles are disabled for this video.")
    except VideoUnavailable:    raise HTTPException(404, "Video not found.")
    except Exception as e:      raise HTTPException(500, f"Transcript error: {e}")

    if len(data) < 5:
        raise HTTPException(400, "Transcript too short to generate chapters.")

    # Build condensed transcript with timestamps (every ~30s snapshot)
    total_dur = data[-1]["start"] + data[-1].get("duration", 0)
    step      = max(30, int(total_dur / 40))   # ~30–40 chapters max

    sampled = []
    last_t  = -step
    for e in data:
        if e["start"] >= last_t + step:
            sampled.append({"t": int(e["start"]), "text": e["text"]})
            last_t = e["start"]

    snapshot_text = "\n".join(f"[{fmt_ts(s['t'])}] {s['text']}" for s in sampled)
    total_str     = fmt_ts(total_dur)

    result = await call_claude(
        "You are a YouTube video chapter specialist. Analyze transcripts and generate clean, descriptive chapter timestamps.",
        f"""Analyze this YouTube transcript and generate chapter timestamps for YouTube.

Total video duration: {total_str}
Transcript snapshots:
{snapshot_text}

Rules:
- First chapter MUST start at 0:00
- Generate 5–12 chapters depending on content variety
- Each chapter title should be concise (3–6 words), descriptive, and engaging
- Timestamps must be in MM:SS or H:MM:SS format
- Space chapters evenly across the video

Output JSON only:
{{
  "chapters": [
    {{"timestamp": "0:00", "seconds": 0, "title": "Introduction"}},
    {{"timestamp": "1:24", "seconds": 84, "title": "Chapter Title Here"}}
  ],
  "youtube_format": "0:00 Introduction\\n1:24 Chapter Title Here\\n..."
}}

Output JSON only, no other text.""", 1000)

    try:
        parsed = parse_json_safe(result)
        return {
            "video_id":      vid,
            "total_duration": total_str,
            "chapter_count":  len(parsed.get("chapters", [])),
            "chapters":       parsed.get("chapters", []),
            "youtube_format": parsed.get("youtube_format", ""),
        }
    except Exception:
        raise HTTPException(500, "Failed to parse AI response. Please try again.")

# ── Metadata & Health ────────────────────────────
@app.get("/api/metadata")
def get_metadata(url: str=Query(...)):
    try:
        with yt_dlp.YoutubeDL({"quiet":True,"skip_download":True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return {"video_id":info.get("id"),"title":info.get("title"),
                "description":(info.get("description") or "")[:600],
                "channel":info.get("uploader"),"channel_url":info.get("uploader_url"),
                "view_count":info.get("view_count"),"like_count":info.get("like_count"),
                "upload_date":info.get("upload_date"),"duration":info.get("duration"),
                "tags":(info.get("tags") or [])[:30],"categories":info.get("categories") or [],
                "thumbnail":info.get("thumbnail")}
    except Exception as e: raise HTTPException(500, f"Metadata error: {e}")

@app.get("/api/health")
def health():
    return {"status":"ok","version":"4.0.0","ai_enabled":bool(ANTHROPIC_KEY),
            "supported_languages":len(LANGUAGES),
            "features":["transcript","summary","thumbnail","tags","title","channel","translate","chapters","metadata"]}
