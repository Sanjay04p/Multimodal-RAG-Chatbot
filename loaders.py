import os
import gc
import streamlit as st
from pathlib import Path
import pypdf
import pytesseract
from PIL import Image
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Model Caching (Crucial for Streamlit Cloud) ---
@st.cache_resource(show_spinner="Loading AI Models into memory...")
def load_models():
    w_model = whisper.load_model("base")
    b_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    b_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return w_model, b_processor, b_model

# --- Helper: Visual Captioning ---
def caption_image(image, processor, model):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# --- Text & PDF Loaders ---
def load_text_files(folder):
    texts = []
    for p in Path(folder).rglob("*.txt"):
        texts.append(p.read_text(errors="ignore"))
    return texts

def load_pdfs(folder):
    texts = []
    for p in Path(folder).rglob("*.pdf"):
        reader = pypdf.PdfReader(str(p))
        content = [page.extract_text() or "" for page in reader.pages]
        texts.append("\n".join(content))
    return texts

# --- Image Loader (OCR + Caption) ---
def image_to_text(img_path):
    _, b_processor, b_model = load_models()
    img = Image.open(img_path).convert('RGB')
    
    ocr_text = pytesseract.image_to_string(img).strip()
    visual_caption = caption_image(img, b_processor, b_model)
    
    # Free up memory
    img.close()
    
    combined_text = f"--- Image Document ---\nVisual Description: {visual_caption}\n"
    if ocr_text:
        combined_text += f"Extracted Text (OCR): {ocr_text}\n"
        
    return combined_text

def load_images(folder):
    texts = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for p in Path(folder).rglob(ext):
            texts.append(image_to_text(str(p)))
    return texts

# --- Video Loader (Audio + Frame Captions) ---
def extract_and_caption_frames(video_path, frame_interval_seconds=10):
    _, b_processor, b_model = load_models()
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    
    success, image = vidcap.read()
    count = 0
    captions = []
    
    while success:
        # Extract 1 frame per 'frame_interval_seconds' (Increased to 10s to save RAM)
        if count % int(fps * frame_interval_seconds) == 0:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_image)
            
            caption = caption_image(pil_img, b_processor, b_model)
            timestamp = int(count / fps)
            captions.append(f"[Frame at {timestamp}s]: {caption}")
            
            # Explicitly delete image arrays from memory
            del rgb_image, pil_img
            
        success, image = vidcap.read()
        count += 1
        
    vidcap.release()
    return "\n".join(captions)

def video_to_text(video_path):
    w_model, _, _ = load_models()
    audio_path = f"temp_audio_{os.path.basename(video_path)}.wav"
    
    try:
        # 1. Audio Transcription
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None)
        transcript_result = w_model.transcribe(audio_path)
        audio_text = transcript_result["text"]
        clip.close()
        
        # 2. Visual Frame Captioning
        visual_text = extract_and_caption_frames(video_path, frame_interval_seconds=10)
        
        return (
            f"--- Video Document ---\n"
            f"Audio Transcription:\n{audio_text}\n\n"
            f"Visual Event Timeline:\n{visual_text}\n"
        )
    finally:
        # 3. Aggressive Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)
        gc.collect()

def load_videos(folder):
    texts = []
    for p in Path(folder).rglob("*.mp4"):
        texts.append(video_to_text(str(p)))
    return texts

# --- Master Builder ---
def build_corpus(base_folder):
    collection = []
    collection += load_text_files(base_folder)
    collection += load_pdfs(base_folder)
    collection += load_images(base_folder)
    collection += load_videos(base_folder)
    return collection