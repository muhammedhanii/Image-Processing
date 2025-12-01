"""
AIU Image Toolkit Backend - Flask REST API
Exposes all image processing functions as REST endpoints.
"""

import os
import io
import base64
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from PIL import Image

from image_processor import (
    # Core utilities
    to_grayscale,
    calc_psnr,
    calc_correlation,
    # Spatial enhancement
    negative_transform,
    log_transform,
    gamma_transform,
    hist_equalization,
    binary_segmentation,
    # Noise filters
    add_salt_pepper,
    apply_gaussian_filter,
    apply_median_filter,
    # Frequency domain
    get_frequency_spectrum,
    apply_ideal_lpf,
    apply_ideal_hpf,
    apply_notch_filter,
    # Cryptography
    xor_encrypt_decrypt
)

app = Flask(__name__)
CORS(app)  # Enable CORS for Next.js frontend

# Configure maximum upload size (16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to NumPy array."""
    # Remove data URL prefix if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    
    # Decode base64
    image_bytes = base64.b64decode(image_data)
    
    # Convert to PIL Image then to NumPy array
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)


def encode_image(image_array: np.ndarray, format: str = 'PNG') -> str:
    """Encode NumPy array to base64 string."""
    # Convert to PIL Image
    if len(image_array.shape) == 2:
        # Grayscale
        image = Image.fromarray(image_array.astype(np.uint8), mode='L')
    else:
        # Color
        image = Image.fromarray(image_array.astype(np.uint8))
    
    # Convert to bytes
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    
    # Encode to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/{format.lower()};base64,{image_base64}"


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'AIU Image Toolkit API is running'})


# ==============================================================================
# Core Utilities Endpoints
# ==============================================================================

@app.route('/api/to-grayscale', methods=['POST'])
def api_to_grayscale():
    """Convert RGB image to grayscale."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        
        result = to_grayscale(image_array)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/calc-psnr', methods=['POST'])
def api_calc_psnr():
    """Calculate PSNR between two images."""
    try:
        data = request.get_json()
        original = decode_image(data['original'])
        decrypted = decode_image(data['decrypted'])
        
        psnr_value = calc_psnr(original, decrypted)
        
        return jsonify({
            'success': True,
            'psnr': psnr_value
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/calc-correlation', methods=['POST'])
def api_calc_correlation():
    """Calculate correlation between two images."""
    try:
        data = request.get_json()
        original = decode_image(data['original'])
        decrypted = decode_image(data['decrypted'])
        
        correlation_value = calc_correlation(original, decrypted)
        
        return jsonify({
            'success': True,
            'correlation': correlation_value
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ==============================================================================
# Spatial Enhancement Endpoints
# ==============================================================================

@app.route('/api/negative-transform', methods=['POST'])
def api_negative_transform():
    """Apply image negative transformation."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = to_grayscale(image_array)
        
        result = negative_transform(image_array)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/log-transform', methods=['POST'])
def api_log_transform():
    """Apply logarithmic transformation."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = to_grayscale(image_array)
        
        result = log_transform(image_array)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/gamma-transform', methods=['POST'])
def api_gamma_transform():
    """Apply gamma (power-law) transformation."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        gamma = float(data.get('gamma', 1.0))
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = to_grayscale(image_array)
        
        result = gamma_transform(image_array, gamma)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/hist-equalization', methods=['POST'])
def api_hist_equalization():
    """Apply histogram equalization."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = to_grayscale(image_array)
        
        result = hist_equalization(image_array)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/binary-segmentation', methods=['POST'])
def api_binary_segmentation():
    """Apply binary segmentation."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        threshold = float(data.get('threshold', 0.5))
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            image_array = to_grayscale(image_array)
        
        result = binary_segmentation(image_array, threshold)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ==============================================================================
# Noise Filter Endpoints
# ==============================================================================

@app.route('/api/add-salt-pepper', methods=['POST'])
def api_add_salt_pepper():
    """Add salt and pepper noise."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        prob = float(data.get('probability', 0.05))
        
        result = add_salt_pepper(image_array, prob)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/gaussian-filter', methods=['POST'])
def api_gaussian_filter():
    """Apply Gaussian filter."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        size = int(data.get('size', 5))
        sigma = float(data.get('sigma', 1.0))
        
        result = apply_gaussian_filter(image_array, size, sigma)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/median-filter', methods=['POST'])
def api_median_filter():
    """Apply median filter."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        size = int(data.get('size', 5))
        
        result = apply_median_filter(image_array, size)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ==============================================================================
# Frequency Domain Endpoints
# ==============================================================================

@app.route('/api/frequency-spectrum', methods=['POST'])
def api_frequency_spectrum():
    """Get frequency spectrum of image."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        
        result = get_frequency_spectrum(image_array)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ideal-lpf', methods=['POST'])
def api_ideal_lpf():
    """Apply ideal low-pass filter."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        cutoff_ratio = float(data.get('cutoff_ratio', 0.3))
        
        result = apply_ideal_lpf(image_array, cutoff_ratio)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/ideal-hpf', methods=['POST'])
def api_ideal_hpf():
    """Apply ideal high-pass filter."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        cutoff_ratio = float(data.get('cutoff_ratio', 0.1))
        
        result = apply_ideal_hpf(image_array, cutoff_ratio)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/notch-filter', methods=['POST'])
def api_notch_filter():
    """Apply notch filter."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        centers = data.get('centers', [(30, 30)])  # List of (u, v) tuples
        radius = int(data.get('radius', 10))
        
        # Convert centers to list of tuples
        centers = [tuple(c) for c in centers]
        
        result = apply_notch_filter(image_array, centers, radius)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ==============================================================================
# Cryptography Endpoints
# ==============================================================================

@app.route('/api/xor-encrypt', methods=['POST'])
def api_xor_encrypt():
    """Encrypt image using XOR."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        key = int(data.get('key', 123))
        
        result = xor_encrypt_decrypt(image_array, key)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/xor-decrypt', methods=['POST'])
def api_xor_decrypt():
    """Decrypt image using XOR (same as encrypt)."""
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        key = int(data.get('key', 123))
        
        result = xor_encrypt_decrypt(image_array, key)
        
        return jsonify({
            'success': True,
            'result': encode_image(result)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ==============================================================================
# Combined Attack-Decryption Scenario Endpoint
# ==============================================================================

@app.route('/api/attack-scenario', methods=['POST'])
def api_attack_scenario():
    """
    Run the complete attack-decryption scenario:
    1. Encrypt the image
    2. Add noise (simulated attack)
    3. Decrypt the attacked image
    4. Calculate PSNR and correlation
    """
    try:
        data = request.get_json()
        image_array = decode_image(data['image'])
        key = int(data.get('key', 123))
        noise_prob = float(data.get('noise_probability', 0.02))
        
        # Convert to grayscale for consistent processing
        if len(image_array.shape) == 3:
            original = to_grayscale(image_array)
        else:
            original = image_array
        
        # Step 1: Encrypt
        encrypted = xor_encrypt_decrypt(original, key)
        
        # Step 2: Attack (add noise)
        attacked = add_salt_pepper(encrypted, noise_prob)
        
        # Step 3: Decrypt
        decrypted = xor_encrypt_decrypt(attacked, key)
        
        # Step 4: Calculate metrics
        psnr_value = calc_psnr(original, decrypted)
        correlation_value = calc_correlation(original, decrypted)
        
        return jsonify({
            'success': True,
            'original': encode_image(original),
            'encrypted': encode_image(encrypted),
            'attacked': encode_image(attacked),
            'decrypted': encode_image(decrypted),
            'psnr': psnr_value,
            'correlation': correlation_value
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ==============================================================================
# API Documentation Endpoint
# ==============================================================================

@app.route('/api/docs', methods=['GET'])
def api_docs():
    """Return API documentation."""
    docs = {
        'name': 'AIU Image Toolkit API',
        'version': '1.0.0',
        'description': 'Image Processing and Cryptography Backend based on CSE281 Lectures',
        'endpoints': {
            'Core Utilities': {
                'POST /api/to-grayscale': {
                    'description': 'Convert RGB image to grayscale',
                    'body': {'image': 'base64 encoded image'}
                },
                'POST /api/calc-psnr': {
                    'description': 'Calculate PSNR between two images',
                    'body': {'original': 'base64 image', 'decrypted': 'base64 image'}
                },
                'POST /api/calc-correlation': {
                    'description': 'Calculate correlation between two images',
                    'body': {'original': 'base64 image', 'decrypted': 'base64 image'}
                }
            },
            'Spatial Enhancement': {
                'POST /api/negative-transform': {
                    'description': 'Apply image negative (S = 255 - r)',
                    'body': {'image': 'base64 image'}
                },
                'POST /api/log-transform': {
                    'description': 'Apply logarithmic transformation',
                    'body': {'image': 'base64 image'}
                },
                'POST /api/gamma-transform': {
                    'description': 'Apply gamma/power-law transformation',
                    'body': {'image': 'base64 image', 'gamma': 'float (default: 1.0)'}
                },
                'POST /api/hist-equalization': {
                    'description': 'Apply histogram equalization',
                    'body': {'image': 'base64 image'}
                },
                'POST /api/binary-segmentation': {
                    'description': 'Convert to binary image',
                    'body': {'image': 'base64 image', 'threshold': 'float 0-1 (default: 0.5)'}
                }
            },
            'Noise Filters': {
                'POST /api/add-salt-pepper': {
                    'description': 'Add salt & pepper noise',
                    'body': {'image': 'base64 image', 'probability': 'float (default: 0.05)'}
                },
                'POST /api/gaussian-filter': {
                    'description': 'Apply Gaussian smoothing filter',
                    'body': {'image': 'base64 image', 'size': 'int (default: 5)', 'sigma': 'float (default: 1.0)'}
                },
                'POST /api/median-filter': {
                    'description': 'Apply median filter',
                    'body': {'image': 'base64 image', 'size': 'int (default: 5)'}
                }
            },
            'Frequency Domain': {
                'POST /api/frequency-spectrum': {
                    'description': 'Get DFT magnitude spectrum',
                    'body': {'image': 'base64 image'}
                },
                'POST /api/ideal-lpf': {
                    'description': 'Apply ideal low-pass filter',
                    'body': {'image': 'base64 image', 'cutoff_ratio': 'float (default: 0.3)'}
                },
                'POST /api/ideal-hpf': {
                    'description': 'Apply ideal high-pass filter',
                    'body': {'image': 'base64 image', 'cutoff_ratio': 'float (default: 0.1)'}
                },
                'POST /api/notch-filter': {
                    'description': 'Apply notch filter',
                    'body': {'image': 'base64 image', 'centers': '[[u,v],...] (default: [[30,30]])', 'radius': 'int (default: 10)'}
                }
            },
            'Cryptography': {
                'POST /api/xor-encrypt': {
                    'description': 'Encrypt image using XOR',
                    'body': {'image': 'base64 image', 'key': 'int 0-255 (default: 123)'}
                },
                'POST /api/xor-decrypt': {
                    'description': 'Decrypt image using XOR',
                    'body': {'image': 'base64 image', 'key': 'int 0-255 (default: 123)'}
                }
            },
            'Combined Scenario': {
                'POST /api/attack-scenario': {
                    'description': 'Run complete attack-decryption scenario',
                    'body': {
                        'image': 'base64 image',
                        'key': 'int 0-255 (default: 123)',
                        'noise_probability': 'float (default: 0.02)'
                    }
                }
            }
        }
    }
    return jsonify(docs)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
