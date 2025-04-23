import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
from transformers import AutoProcessor, WhisperForConditionalGeneration, pipeline
import nltk
nltk.data.path.append("./utils")
nltk.download('punkt_tab', download_dir="./utils")
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import argparse
import logging
import time

def mp3_2_waveform(mp3_path, target_sr = 16000):
    info = torchaudio.info(mp3_path)
    
    waveform, sample_rate = torchaudio.load(mp3_path, num_frames=info.num_frames)
    duration = waveform.shape[1] / sample_rate

    if sample_rate != target_sr: #resample to 16k
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    if waveform.shape[0] > 1: #mono
        waveform = torch.mean(waveform, dim=0)

    return waveform, sample_rate, duration

def pipeline_inference(pipe, waveform, sample_rate): 
    result = pipe({"array": waveform.numpy().squeeze(), "sampling_rate": sample_rate})
    
    return result["text"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default=os.environ.get("DATA_FOLDER", "../large_data"))
    parser.add_argument('--model_name', type=str, default=os.environ.get("MODEL_NAME", "openai/whisper-large-v3-turbo"))
    parser.add_argument('--log_file', type=str, default=os.environ.get("LOG_NAME", "inference.log"))
    args = parser.parse_args()

    file_handler = logging.FileHandler(args.log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    logger = logging.getLogger()

    logger.info(f"Loading model and processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    ).to("cuda")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device="cuda",
        return_timestamps=True
    )

    logger.info(f"Starting transcription in folder: {args.model_name}")
    
    for root, dirs, files in tqdm(
        os.walk(args.data_folder),
        desc="Directories",
        file=file_handler.stream
    ):
        if len(dirs) == 0 and ".ipynb_checkpoints" not in root and "__MACOSX" not in root:
            for file in tqdm(
                files,
                desc=f"Files in {os.path.basename(root)}",
                leave=False,
                file=file_handler.stream
            ):
                if file.endswith(".mp3"):
                    mp3_path = os.path.join(root, file)
                    txt_path = os.path.join(root, os.path.splitext(file)[0] + '.txt')
                    logger.info(f"Processing: {mp3_path}")
                    waveform, sample_rate, duration = mp3_2_waveform(mp3_path, target_sr=16000)
                    logger.info(f"Duration: {duration:.2f} sec")

                    transcript = pipeline_inference(pipe, waveform, sample_rate)

                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(transcript)
                    logger.info(f"Saved TXT: {txt_path}")

                    sentences = sent_tokenize(transcript)
                    csv_path = txt_path.replace(".txt", ".csv")

                    pd.DataFrame({
                        "sentence_id": range(1, len(sentences) + 1),
                        "sentence": sentences
                    }).to_csv(csv_path, index=False)
                    logger.info(f"Saved CSV: {csv_path}")

    logger.info("✅ Transcription complete.")
