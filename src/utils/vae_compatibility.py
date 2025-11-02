"""
VAE Compatibility Validation Module
Validates shape compatibility between VAE models and the diffusion pipeline
"""

import torch
from typing import Optional, Dict, Any, Tuple
from safetensors.torch import safe_open

def validate_vae_compatibility(vae_path: str, 
                               expected_latent_channels: int = 16,
                               expected_spatial_downsample: int = 8,
                               expected_temporal_downsample: int = 4,
                               debug: Optional[Any] = None) -> Tuple[bool, str]:
    """
    Validate VAE model compatibility with the pipeline
    
    Args:
        vae_path: Path to VAE safetensors file
        expected_latent_channels: Expected latent space channels (default: 16)
        expected_spatial_downsample: Expected spatial downsampling factor (default: 8)
        expected_temporal_downsample: Expected temporal downsampling factor (default: 4)
        debug: Optional debug instance for logging
        
    Returns:
        Tuple of (is_compatible: bool, message: str)
    """
    try:
        # Load VAE weights to inspect shapes
        with safe_open(vae_path, framework='pt', device='cpu') as f:
            keys = list(f.keys())
            
            # Check for encoder/decoder structure
            encoder_keys = [k for k in keys if 'encoder' in k.lower()]
            decoder_keys = [k for k in keys if 'decoder' in k.lower()]
            
            if not encoder_keys or not decoder_keys:
                return False, f"VAE structure validation failed: Missing encoder ({len(encoder_keys)}) or decoder ({len(decoder_keys)}) layers"
            
            # Check for quant_conv layers to determine latent channels
            quant_conv_keys = [k for k in keys if 'quant_conv' in k.lower() and 'weight' in k]
            
            if quant_conv_keys:
                # Get the first quant_conv weight to check output channels
                quant_conv_weight = f.get_tensor(quant_conv_keys[0])
                
                # For Conv layers, shape is typically [out_channels, in_channels, ...]
                out_channels = quant_conv_weight.shape[0]
                
                # For VAE, quant_conv typically outputs 2*latent_channels (mean and logvar)
                # So latent_channels = out_channels / 2
                if out_channels % 2 == 0:
                    inferred_latent_channels = out_channels // 2
                else:
                    inferred_latent_channels = out_channels
                
                if inferred_latent_channels != expected_latent_channels:
                    msg = (f"VAE latent channels mismatch: Expected {expected_latent_channels}, "
                          f"found {inferred_latent_channels} (quant_conv output: {out_channels})")
                    if debug:
                        debug.log(msg, level="WARNING", category="vae", force=True)
                    return False, msg
            
            # Check decoder output layer for channel consistency
            decoder_out_keys = [k for k in keys if 'decoder' in k.lower() and ('conv_out' in k or 'final' in k) and 'weight' in k]
            
            if decoder_out_keys:
                decoder_out_weight = f.get_tensor(decoder_out_keys[0])
                # Conv output should be 3 channels for RGB output
                out_channels = decoder_out_weight.shape[0]
                if out_channels != 3:
                    msg = f"VAE decoder output channels unexpected: Expected 3 (RGB), found {out_channels}"
                    if debug:
                        debug.log(msg, level="WARNING", category="vae", force=True)
            
            # Log basic statistics
            if debug:
                debug.log(f"VAE validation passed:", category="vae", force=True)
                debug.log(f"  - Total parameters: {len(keys)}", category="vae")
                debug.log(f"  - Encoder layers: {len(encoder_keys)}", category="vae")
                debug.log(f"  - Decoder layers: {len(decoder_keys)}", category="vae")
                debug.log(f"  - Latent channels: {expected_latent_channels} (validated)", category="vae")
            
            return True, "VAE compatibility validated successfully"
            
    except Exception as e:
        msg = f"VAE validation error: {str(e)}"
        if debug:
            debug.log(msg, level="ERROR", category="vae", force=True)
        return False, msg


def get_vae_info(vae_path: str, debug: Optional[Any] = None) -> Dict[str, Any]:
    """
    Extract VAE model information from safetensors file
    
    Args:
        vae_path: Path to VAE safetensors file
        debug: Optional debug instance for logging
        
    Returns:
        Dictionary with VAE information including latent_channels, total_params, etc.
    """
    info = {
        'latent_channels': None,
        'total_params': 0,
        'encoder_layers': 0,
        'decoder_layers': 0,
        'has_quant_conv': False,
        'compatible': False,
        'error': None
    }
    
    try:
        with safe_open(vae_path, framework='pt', device='cpu') as f:
            keys = list(f.keys())
            info['total_params'] = len(keys)
            
            # Count encoder/decoder layers
            info['encoder_layers'] = len([k for k in keys if 'encoder' in k.lower()])
            info['decoder_layers'] = len([k for k in keys if 'decoder' in k.lower()])
            
            # Check for quant_conv
            quant_conv_keys = [k for k in keys if 'quant_conv' in k.lower() and 'weight' in k]
            info['has_quant_conv'] = len(quant_conv_keys) > 0
            
            if quant_conv_keys:
                quant_conv_weight = f.get_tensor(quant_conv_keys[0])
                out_channels = quant_conv_weight.shape[0]
                if out_channels % 2 == 0:
                    info['latent_channels'] = out_channels // 2
                else:
                    info['latent_channels'] = out_channels
        
        info['compatible'] = True
        
    except Exception as e:
        info['error'] = str(e)
        if debug:
            debug.log(f"Error extracting VAE info: {e}", level="ERROR", category="vae", force=True)
    
    return info
