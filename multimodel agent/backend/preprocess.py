import os
import re
import pandas as pd
import numpy as np
import librosa

def preprocess_subject(participant_id, data_dir, cache_dir, fixed_audio_len=500):
    """
    Extracts acoustic and linguistic features for a single participant.
    """
    subject_dir = os.path.join(data_dir, f"{participant_id}_P")
    if not os.path.isdir(subject_dir):
        print(f"[SKIP] Directory not found for subject: {participant_id} (Path: {subject_dir})")
        return False

    audio_path = os.path.join(subject_dir, f"{participant_id}_AUDIO.wav")
    transcript_path = os.path.join(subject_dir, f"{participant_id}_TRANSCRIPT.csv")

    if not os.path.exists(audio_path):
        print(f"[SKIP] Missing audio file for subject: {participant_id}")
        return False
    if not os.path.exists(transcript_path):
        print(f"[SKIP] Missing transcript file for subject: {participant_id}")
        return False

    print(f"[PROCESS] Processing Participant {participant_id}...")

    # 1. Acoustic Features: Load WAV, resample to 16kHz, extract Mel-spec & MFCC
    try:
        # Load audio (mono)
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Extract Mel-Spectrogram (128 channels)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)  # shape: [128, time]
        
        # Extract 13 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # shape: [13, time]
        
        # Concatenate features along the frequency/descriptor axis
        # Resulting shape: [time_steps, 141] (128 mel + 13 mfcc)
        audio_features = np.concatenate([mel_db, mfccs], axis=0).T
        
        # Pad or truncate to a fixed length (e.g. 500 frames)
        time_steps = audio_features.shape[0]
        if time_steps < fixed_audio_len:
            pad_width = ((0, fixed_audio_len - time_steps), (0, 0))
            audio_features = np.pad(audio_features, pad_width, mode='constant', constant_values=0.0)
        else:
            audio_features = audio_features[:fixed_audio_len, :]
            
    except Exception as e:
        print(f"[ERROR] Failed to extract acoustic features for subject {participant_id}: {str(e)}")
        return False

    # 2. Linguistic Features: Parse participant responses from the transcript CSV
    try:
        # Transcripts are typically tab-separated in DAIC-WOZ
        df_trans = pd.read_csv(transcript_path, sep='\t')
        
        # Standardize column headers
        df_trans.columns = [col.lower().strip() for col in df_trans.columns]
        
        # Filter rows spoken by the "Participant" (ignoring virtual interviewer "Ellie")
        speaker_col = 'speaker' if 'speaker' in df_trans.columns else 'speaker_id'
        text_col = 'value' if 'value' in df_trans.columns else 'text'
        
        if speaker_col not in df_trans.columns or text_col not in df_trans.columns:
            raise KeyError(f"Transcript column mismatch. Available keys: {list(df_trans.columns)}")
            
        participant_df = df_trans[df_trans[speaker_col].str.lower() == 'participant']
        utterances = participant_df[text_col].dropna().tolist()
        
        # Mock embeddings to output shape [num_utterances, 768] (whisper/ClinicalBERT space)
        # Note: In production, replace this block with actual model inference:
        # text_features = sentence_transformer_model.encode(utterances)
        num_utterances = len(utterances)
        text_dim = 768
        if num_utterances == 0:
            text_features = np.zeros((1, text_dim), dtype=np.float32)
        else:
            text_features = np.zeros((num_utterances, text_dim), dtype=np.float32)
            # Encode character lengths/word counts dynamically to preserve variance
            for i, utt in enumerate(utterances):
                word_count = len(re.findall(r'\b\w+\b', utt))
                text_features[i, :10] = word_count * 0.05  # populate initial dimensions
                
    except Exception as e:
        print(f"[ERROR] Failed to parse linguistic features for subject {participant_id}: {str(e)}")
        return False

    # 3. Save processed arrays in cache sub-directories
    try:
        audio_cache_path = os.path.join(cache_dir, 'audio', f"{participant_id}.npy")
        text_cache_path = os.path.join(cache_dir, 'text', f"{participant_id}.npy")
        
        np.save(audio_cache_path, audio_features)
        np.save(text_cache_path, text_features)
        
        print(f"[SUCCESS] Subject {participant_id} features cached successfully. "
              f"Audio: {audio_features.shape}, Text: {text_features.shape}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write cache files for subject {participant_id}: {str(e)}")
        return False


def run_preprocessing():
    # Setup directories
    data_dir = './data'
    cache_dir = os.path.join(data_dir, 'cache')
    
    os.makedirs(os.path.join(cache_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, 'text'), exist_ok=True)
    
    # Process both train and dev splits
    splits = [
        'train_split_Depression_AVEC2017.csv',
        'dev_split_Depression_AVEC2017.csv'
    ]
    
    processed_count = 0
    skipped_count = 0

    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            print(f"[Warning] Split CSV not found at: {split_path}")
            continue

        print(f"\n--- Loading split: {split} ---")
        df = pd.read_csv(split_path)
        
        # Standardize subject column name
        subject_col = None
        for col in ['Participant_ID', 'ID', 'subject_id']:
            if col in df.columns:
                subject_col = col
                break
                
        if not subject_col:
            print(f"[ERROR] Identifier key column missing in CSV: {split_path}")
            continue

        for participant_id in df[subject_col].unique():
            success = preprocess_subject(participant_id, data_dir, cache_dir)
            if success:
                processed_count += 1
            else:
                skipped_count += 1

    print(f"\n--- Preprocessing completed! ---")
    print(f"Total processed and cached: {processed_count}")
    print(f"Total skipped: {skipped_count}\n")


if __name__ == "__main__":
    run_preprocessing()
