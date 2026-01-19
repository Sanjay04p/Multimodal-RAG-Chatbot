from pathlib import Path
import pypdf

import pytesseract
from PIL import Image
import whisper
from moviepy.editor import VideoFileClip

def load_text_files(folder):
    texts = []
    for p in Path(folder).rglob("*.txt"):
        texts.append(p.read_text(errors="ignore"))
    return texts

def load_pdfs(folder):
    texts = []
    for p in Path(folder).rglob("*.pdf"):
        reader = pypdf.PdfReader(str(p))
        content = []
        for page in reader.pages:
            content.append(page.extract_text() or "")
        texts.append("\n".join(content))
    return texts

def image_to_text(img_path):
    img = Image.open(img_path)
    return pytesseract.image_to_string(img)

def load_images(folder):
    texts = []
    for p in Path(folder).rglob("*.png"):
        texts.append(image_to_text(str(p)))
    for p in Path(folder).rglob("*.jpg"):
        texts.append(image_to_text(str(p)))
    return texts


whisper_model = whisper.load_model("base")
def video_to_text(video_path):
    clip = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
    result = whisper_model.transcribe(audio_path)
    clip.close() 
    return result["text"]

def load_videos(folder):
    texts = []
    for p in Path(folder).rglob("*.mp4"):
        texts.append(video_to_text(str(p)))
    return texts

def build_corpus(base_folder):
    collection = []
    collection += load_text_files(base_folder)
    collection += load_pdfs(base_folder)
    collection += load_images(base_folder)
    collection += load_videos(base_folder)
    return collection
