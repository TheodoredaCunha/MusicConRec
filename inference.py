"""
Utility functions for loading pre-trained MusicConRec models and inference
"""

import torch
import torchaudio
from model.model import MusicConRec
import os


class MusicConRecInference:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Load a pre-trained MusicConRec model for inference
        
        Args:
            checkpoint_path: Path to saved model weights (.pth file)
            device: "cuda" or "cpu"
        """
        self.device = device
        self.model = MusicConRec().to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        print(f"✓ Model loaded from: {checkpoint_path}")
    
    def get_audio_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Get audio embedding (z_audio)
        
        Args:
            audio: (1, 1, T) audio tensor
            
        Returns:
            (1, 128) embedding tensor
        """
        audio = audio.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(audio, torch.zeros(1, 1, 13).to(self.device))
            z_audio = outputs['z_audio']  # (B, 128)
        
        return z_audio
    
    def get_chord_embedding(self, chord: torch.Tensor) -> torch.Tensor:
        """
        Get chord-beat embedding (z_chord)
        
        Args:
            chord: (1, T, 13) chord-beat tensor
            
        Returns:
            (1, 128) embedding tensor
        """
        chord = chord.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(torch.zeros(1, 1, 8000).to(self.device), chord)
            z_chord = outputs['z_chord']  # (B, 128)
        
        return z_chord
    
    def get_audio_representation(self, audio: torch.Tensor) -> dict:
        """
        Get full audio representation (embedding + hidden state)
        
        Args:
            audio: (1, 1, T) audio tensor
            
        Returns:
            dict with 'z_audio' (128,) and 'h_audio' (128,) tensors
        """
        audio = audio.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(audio, torch.zeros(1, 1, 13).to(self.device))
            return {
                'z_audio': outputs['z_audio'].squeeze(0),  # (128,)
                'h_audio': outputs['h_audio'].squeeze(0)   # (128,)
            }
    
    def reconstruct_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct audio from input
        
        Args:
            audio: (1, 1, T) audio tensor
            
        Returns:
            (1, 1, T) reconstructed audio tensor
        """
        audio = audio.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(audio, torch.zeros(1, 1, 13).to(self.device))
            x_recon = outputs['x_recon']  # (B, 1, T)
        
        return x_recon.cpu()


def load_best_model(model_dir: str = "./outputs", device: str = "cuda") -> MusicConRecInference:
    """
    Load the best saved model from training
    
    Args:
        model_dir: Directory containing saved models
        device: "cuda" or "cpu"
        
    Returns:
        MusicConRecInference instance
    """
    best_model_path = os.path.join(model_dir, "best_model.pth")
    
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at: {best_model_path}")
    
    return MusicConRecInference(best_model_path, device)


def load_final_model(model_dir: str = "./outputs", device: str = "cuda") -> MusicConRecInference:
    """
    Load the final saved model from training
    
    Args:
        model_dir: Directory containing saved models
        device: "cuda" or "cpu"
        
    Returns:
        MusicConRecInference instance
    """
    final_model_path = os.path.join(model_dir, "final_model.pth")
    
    if not os.path.exists(final_model_path):
        raise FileNotFoundError(f"Final model not found at: {final_model_path}")
    
    return MusicConRecInference(final_model_path, device)


def inference():
    # Load model
    model = load_best_model()

    # Get device from model
    device = next(model.model.parameters()).device

    # Load audio file
    # torchaudio already converts PCM -> float32 in [-1, 1]
    audio, sr = torchaudio.load("./inference/audio_sample.wav")

    print(f"Original audio shape: {audio.shape}")
    print(f"Sample rate: {sr}")

    # Add batch dimension
    # (C, T) -> (1, C, T)
    audio = audio.unsqueeze(0).to(device)

    # -----------------------------
    # Generate embedding
    # -----------------------------
    with torch.no_grad():
        embedding = model.get_audio_embedding(audio)

    print(f"Audio embedding shape: {embedding.shape}")

    # -----------------------------
    # Reconstruct audio
    # -----------------------------
    with torch.no_grad():
        recon_audio = model.reconstruct_audio(audio)

    print(f"Reconstructed audio shape: {recon_audio.shape}")

    # Remove batch dimension
    # (1, C, T) -> (C, T)
    recon_audio = recon_audio.squeeze(0)

    # Clamp output to valid audio range
    recon_audio = recon_audio.clamp(-1.0, 1.0)

    # Move to CPU for saving
    recon_audio = recon_audio.cpu()

    print(f"Reconstruction min: {recon_audio.min().item():.4f}")
    print(f"Reconstruction max: {recon_audio.max().item():.4f}")

    # Save reconstructed audio
    torchaudio.save(
        "./inference/reconstructed_audio.wav",
        recon_audio,
        sr
    )

    print("✓ Reconstructed audio saved to reconstructed_audio.wav")
