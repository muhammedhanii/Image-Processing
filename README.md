# AIU Image Toolkit

A comprehensive full-stack application for Image Processing and Cryptography based on CSE281 Lecture Notes.

## Project Structure

```
├── backend/
│   ├── image_processor.py   # Core image processing functions
│   ├── app.py               # Flask REST API
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/app/page.tsx     # Next.js frontend application
│   └── ...                  # Next.js project files
└── Lecture_*.pdf            # CSE281 Lecture notes
```

## Features

### Core Utilities
- **to_grayscale**: Convert RGB image to grayscale
- **calc_psnr**: Calculate Peak Signal-to-Noise Ratio between two images
- **calc_correlation**: Calculate correlation coefficient between two images

### Spatial Domain Enhancement
- **negative_transform**: Image negative (S = 255 - r)
- **log_transform**: Logarithmic transformation (s = c * ln(1 + r))
- **gamma_transform**: Power-law/gamma transformation (s = c * r^γ)
- **hist_equalization**: Histogram equalization for contrast improvement
- **binary_segmentation**: Binary thresholding

### Spatial Noise Filters
- **add_salt_pepper**: Add Salt & Pepper noise
- **apply_gaussian_filter**: Gaussian smoothing filter
- **apply_median_filter**: Median filter for noise removal

### Frequency Domain Filtering
- **get_frequency_spectrum**: DFT magnitude spectrum visualization
- **apply_ideal_lpf**: Ideal Low-Pass Filter
- **apply_ideal_hpf**: Ideal High-Pass Filter
- **apply_notch_filter**: Notch filter for periodic noise removal

### Cryptography
- **xor_encrypt_decrypt**: XOR-based image encryption/decryption

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Flask server:
   ```bash
   python app.py
   ```
   
   The API will be available at `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create `.env.local` file:
   ```
   NEXT_PUBLIC_API_URL=http://localhost:5000
   ```

4. Run the development server:
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:3000`

## API Endpoints

### Core Utilities
- `POST /api/to-grayscale` - Convert to grayscale
- `POST /api/calc-psnr` - Calculate PSNR
- `POST /api/calc-correlation` - Calculate correlation

### Spatial Enhancement
- `POST /api/negative-transform` - Apply negative transform
- `POST /api/log-transform` - Apply log transform
- `POST /api/gamma-transform` - Apply gamma transform
- `POST /api/hist-equalization` - Apply histogram equalization
- `POST /api/binary-segmentation` - Apply binary segmentation

### Noise Filters
- `POST /api/add-salt-pepper` - Add salt & pepper noise
- `POST /api/gaussian-filter` - Apply Gaussian filter
- `POST /api/median-filter` - Apply median filter

### Frequency Domain
- `POST /api/frequency-spectrum` - Get frequency spectrum
- `POST /api/ideal-lpf` - Apply ideal low-pass filter
- `POST /api/ideal-hpf` - Apply ideal high-pass filter
- `POST /api/notch-filter` - Apply notch filter

### Cryptography
- `POST /api/xor-encrypt` - Encrypt with XOR
- `POST /api/xor-decrypt` - Decrypt with XOR
- `POST /api/attack-scenario` - Run complete attack-decryption scenario

### Documentation
- `GET /api/docs` - Get API documentation
- `GET /api/health` - Health check endpoint

## Attack-Decryption Scenario

The toolkit includes a complete demonstration of the attack-decryption scenario:

1. **Encrypt**: Original image is encrypted using XOR with a secret key
2. **Attack**: Salt & Pepper noise is added to the encrypted image
3. **Decrypt**: The attacked image is decrypted using the same key
4. **Evaluate**: PSNR and correlation metrics measure image quality degradation

## Academic References

All implementations are based on CSE281 Lecture Notes:
- Lecture 4: Image Negatives, Binary Segmentation
- Lecture 5: Grayscale Conversion, Log Transform, Gamma Transform, Histogram Equalization
- Lecture 6: Salt & Pepper Noise, Gaussian Filter, Median Filter
- Lecture 8: DFT, Frequency Filters (LPF, HPF, Notch), XOR Encryption, PSNR, Correlation

## License

Educational use - AIU CSE281 Course Materials
