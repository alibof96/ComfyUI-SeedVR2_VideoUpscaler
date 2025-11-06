# VAE Integration Guide

## Overview

SeedVR2 supports multiple VAE (Variational Autoencoder) models for encoding and decoding video latents. This document describes the available VAE options and how to use them.

## Supported VAE Models

### 1. ema_vae_fp16.safetensors (Default)
- **Source**: numz/SeedVR2_comfyUI on Hugging Face
- **Precision**: FP16
- **Characteristics**: 
  - High quality encoding/decoding
  - Standard VRAM usage
  - Best for quality-focused workflows
- **Use Case**: Production-quality video upscaling

### 2. taew2_1.safetensors (LightVAE)
- **Source**: lightx2v/Autoencoders on Hugging Face
- **Precision**: FP16
- **Characteristics**:
  - Lightweight temporal autoencoder from LightX2V project
  - Faster inference speed
  - Lower memory footprint (~22.6 MB vs larger default VAE)
  - Slightly reduced quality compared to default VAE
- **Use Case**: Fast prototyping, memory-constrained systems, batch processing

## How to Use Custom VAE

### Method 1: Using ComfyUI Node Interface

1. Add the **SeedVR2 Extra Args** node to your workflow
2. In the `vae_model` dropdown, select your desired VAE:
   - `ema_vae_fp16.safetensors` (default)
   - `taew2_1.safetensors` (LightVAE)
3. Connect the Extra Args output to the SeedVR2 Video Upscaler node
4. Run your workflow

The selected VAE will be automatically downloaded on first use if not already present.

### Method 2: Manual Placement

You can also manually place VAE files in the ComfyUI models directory:
```
ComfyUI/models/SEEDVR2/
```

The system will auto-discover VAE files in this directory.

## Technical Details

### Shape Compatibility

The VAE models must be compatible with the following specifications:
- **Latent Channels**: 16
- **Spatial Downsample Factor**: 8x
- **Temporal Downsample Factor**: 4x
- **Input Channels**: 3 (RGB)
- **Output Channels**: 3 (RGB)

### Automatic Validation

The system automatically validates VAE compatibility when loading:
- Checks encoder/decoder structure
- Validates latent space dimensions
- Verifies channel counts
- Logs warnings if incompatibilities are detected

If validation warnings appear but you want to proceed anyway, the system will continue with the loading process. Monitor the output quality to ensure acceptable results.

## Performance Comparison

| VAE Model | Size | Speed | Quality | VRAM Usage | Best For |
|-----------|------|-------|---------|------------|----------|
| ema_vae_fp16.safetensors | Larger | Standard | High | Standard | Quality-focused work |
| taew2_1.safetensors | ~22.6 MB | Faster | Good | Lower | Fast iteration, prototyping |

## Troubleshooting

### VAE fails to download
- Check your internet connection
- Verify the Hugging Face repository is accessible
- Try manually downloading and placing in `ComfyUI/models/SEEDVR2/`

### Compatibility warnings
- Review the validation messages in the console
- Ensure you're using VAE models designed for video processing
- Check that the VAE supports the required latent dimensions

### Quality issues with taew2_1
- The LightVAE model prioritizes speed over quality
- For production work, consider using the default `ema_vae_fp16.safetensors`
- Adjust other pipeline parameters to compensate if needed

## Adding Custom VAE Models

To add your own VAE models:

1. Ensure compatibility with the required specifications (see Technical Details above)
2. Place the `.safetensors` file in `ComfyUI/models/SEEDVR2/`
3. Optionally register in `src/utils/model_registry.py` for proper metadata and hashing:

```python
"your_vae_name.safetensors": ModelInfo(
    repo="your-repo/your-model",
    category="vae",
    precision="fp16",
    sha256="<optional-hash-for-validation>"
),
```

4. Restart ComfyUI to detect the new VAE

## References

- [LightX2V Project](https://github.com/ModelTC/LightX2V)
- [taew2_1.safetensors on Hugging Face](https://huggingface.co/lightx2v/Autoencoders/blob/main/taew2_1.safetensors)
- [ComfyUI-LightVAE](https://github.com/ModelTC/ComfyUI-LightVAE)
