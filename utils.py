import torch
import numpy as np
from audiomentations import Compose, AddGaussianNoise



# Define augmentation
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
])

def encode_audio(file_path, model, sr=24000, apply_augmentation = False):
    import soundfile as sf
    audio, original_sr = sf.read(file_path)

    if len(audio) == 0:
        audio = np.zeros(1)

    if apply_augmentation:
        # Apply augmentation (CPU)
        audio = augment(samples=audio, sample_rate=sr)

    

    # Convert to tensor and move to GPU
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    audio_tensor = audio_tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        # Encode the audio using EnCodec (GPU)
        encoded_frames = model.encode(audio_tensor)

    codes_list = [frame[0] for frame in encoded_frames]  
    codes = torch.cat(codes_list, dim=2)

    codes = codes.squeeze(0).permute(1, 0).long()

    num_quantizers = codes.shape[1]

    for q in range(num_quantizers):
        max_index = codes[:, q].max().item()
        min_index = codes[:, q].min().item()

    return codes

def apply_delay_pattern(codes, codebook_size, num_quantizers):
    
    num_frames, num_quantizers = codes.shape
    
    max_delay = num_quantizers - 1
    
    padding_value = codebook_size  
    
    delayed_codes = torch.full((num_frames + max_delay, num_quantizers), fill_value=padding_value, dtype=codes.dtype)
    
    for q in range(num_quantizers):
        delayed_codes[q:q + num_frames, q] = codes[:, q]
        
    return delayed_codes  # Shape: [num_frames + max_delay, num_quantizers]

def remove_delay_pattern(delayed_codes, num_quantizers):
    
    num_frames = delayed_codes.shape[0] - (num_quantizers - 1)
    
    codes = torch.zeros(num_frames, num_quantizers, dtype=delayed_codes.dtype)
    
    for q in range(num_quantizers):
        codes[:, q] = delayed_codes[q:q + num_frames, q]
        
    return codes  # Shape: [num_frames, num_quantizers]
