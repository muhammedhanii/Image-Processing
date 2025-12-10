"use client";

import { useState, useRef } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

type ProcessingOperation = 
  | "grayscale"
  | "negative"
  | "log"
  | "gamma"
  | "histogram"
  | "binary"
  | "salt-pepper"
  | "gaussian"
  | "median"
  | "spectrum"
  | "lpf"
  | "hpf"
  | "notch"
  | "encrypt"
  | "decrypt"
  | "attack-scenario";

interface OperationConfig {
  name: string;
  endpoint: string;
  params?: { name: string; type: string; default: number; min?: number; max?: number; step?: number }[];
  category: string;
}

const operations: Record<ProcessingOperation, OperationConfig> = {
  grayscale: { name: "To Grayscale", endpoint: "/api/to-grayscale", category: "Core Utilities" },
  negative: { name: "Negative Transform", endpoint: "/api/negative-transform", category: "Spatial Enhancement" },
  log: { name: "Log Transform", endpoint: "/api/log-transform", category: "Spatial Enhancement" },
  gamma: { 
    name: "Gamma Transform", 
    endpoint: "/api/gamma-transform", 
    params: [{ name: "gamma", type: "number", default: 1.0, min: 0.1, max: 5.0, step: 0.1 }],
    category: "Spatial Enhancement"
  },
  histogram: { name: "Histogram Equalization", endpoint: "/api/hist-equalization", category: "Spatial Enhancement" },
  binary: { 
    name: "Binary Segmentation", 
    endpoint: "/api/binary-segmentation", 
    params: [{ name: "threshold", type: "number", default: 0.5, min: 0, max: 1, step: 0.05 }],
    category: "Spatial Enhancement"
  },
  "salt-pepper": { 
    name: "Add Salt & Pepper Noise", 
    endpoint: "/api/add-salt-pepper", 
    params: [{ name: "probability", type: "number", default: 0.05, min: 0, max: 0.5, step: 0.01 }],
    category: "Noise Filters"
  },
  gaussian: { 
    name: "Gaussian Filter", 
    endpoint: "/api/gaussian-filter", 
    params: [
      { name: "size", type: "number", default: 5, min: 3, max: 15, step: 2 },
      { name: "sigma", type: "number", default: 1.0, min: 0.1, max: 5.0, step: 0.1 }
    ],
    category: "Noise Filters"
  },
  median: { 
    name: "Median Filter", 
    endpoint: "/api/median-filter", 
    params: [{ name: "size", type: "number", default: 5, min: 3, max: 15, step: 2 }],
    category: "Noise Filters"
  },
  spectrum: { name: "Frequency Spectrum", endpoint: "/api/frequency-spectrum", category: "Frequency Domain" },
  lpf: { 
    name: "Ideal Low-Pass Filter", 
    endpoint: "/api/ideal-lpf", 
    params: [{ name: "cutoff_ratio", type: "number", default: 0.3, min: 0.01, max: 1.0, step: 0.05 }],
    category: "Frequency Domain"
  },
  hpf: { 
    name: "Ideal High-Pass Filter", 
    endpoint: "/api/ideal-hpf", 
    params: [{ name: "cutoff_ratio", type: "number", default: 0.1, min: 0.01, max: 1.0, step: 0.05 }],
    category: "Frequency Domain"
  },
  notch: { 
    name: "Notch Filter", 
    endpoint: "/api/notch-filter", 
    params: [
      { name: "center_u", type: "number", default: 30, min: -100, max: 100, step: 5 },
      { name: "center_v", type: "number", default: 30, min: -100, max: 100, step: 5 },
      { name: "radius", type: "number", default: 10, min: 1, max: 50, step: 1 }
    ],
    category: "Frequency Domain"
  },
  encrypt: { 
    name: "XOR Encrypt", 
    endpoint: "/api/xor-encrypt", 
    params: [{ name: "key", type: "number", default: 123, min: 0, max: 255, step: 1 }],
    category: "Cryptography"
  },
  decrypt: { 
    name: "XOR Decrypt", 
    endpoint: "/api/xor-decrypt", 
    params: [{ name: "key", type: "number", default: 123, min: 0, max: 255, step: 1 }],
    category: "Cryptography"
  },
  "attack-scenario": { 
    name: "Attack Scenario", 
    endpoint: "/api/attack-scenario", 
    params: [
      { name: "key", type: "number", default: 123, min: 0, max: 255, step: 1 },
      { name: "noise_probability", type: "number", default: 0.02, min: 0, max: 0.2, step: 0.01 }
    ],
    category: "Cryptography"
  },
};

const categories = ["Core Utilities", "Spatial Enhancement", "Noise Filters", "Frequency Domain", "Cryptography"];

export default function Home() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [selectedOperation, setSelectedOperation] = useState<ProcessingOperation>("grayscale");
  const [params, setParams] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<{ psnr?: number; correlation?: number } | null>(null);
  const [attackScenarioResults, setAttackScenarioResults] = useState<{
    original: string;
    encrypted: string;
    attacked: string;
    decrypted: string;
    psnr: number;
    correlation: number;
  } | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setOriginalImage(event.target?.result as string);
        setProcessedImage(null);
        setMetrics(null);
        setAttackScenarioResults(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleOperationChange = (op: ProcessingOperation) => {
    setSelectedOperation(op);
    // Initialize params with defaults
    const config = operations[op];
    const newParams: Record<string, number> = {};
    config.params?.forEach(p => {
      newParams[p.name] = p.default;
    });
    setParams(newParams);
    setError(null);
  };

  const processImage = async () => {
    if (!originalImage) {
      setError("Please upload an image first");
      return;
    }

    setLoading(true);
    setError(null);
    setMetrics(null);
    setAttackScenarioResults(null);

    try {
      const config = operations[selectedOperation];
      let body: Record<string, unknown> = { image: originalImage, ...params };

      // Handle notch filter specially to construct centers array
      if (selectedOperation === "notch") {
        const center_u = params["center_u"] ?? 30;
        const center_v = params["center_v"] ?? 30;
        body = {
          image: originalImage,
          centers: [[center_u, center_v]],
          radius: params["radius"] ?? 10
        };
      }

      const response = await fetch(`${API_BASE_URL}${config.endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (data.success) {
        if (selectedOperation === "attack-scenario") {
          setAttackScenarioResults({
            original: data.original,
            encrypted: data.encrypted,
            attacked: data.attacked,
            decrypted: data.decrypted,
            psnr: data.psnr,
            correlation: data.correlation,
          });
          setProcessedImage(data.decrypted);
        } else {
          setProcessedImage(data.result);
        }
        
        if (data.psnr !== undefined) {
          setMetrics({ psnr: data.psnr, correlation: data.correlation });
        }
      } else {
        setError(data.error || "Processing failed");
      }
    } catch (err) {
      setError(`Failed to process image: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const calculateMetrics = async () => {
    if (!originalImage || !processedImage) {
      setError("Need both original and processed images to calculate metrics");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const [psnrRes, corrRes] = await Promise.all([
        fetch(`${API_BASE_URL}/api/calc-psnr`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ original: originalImage, decrypted: processedImage }),
        }),
        fetch(`${API_BASE_URL}/api/calc-correlation`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ original: originalImage, decrypted: processedImage }),
        }),
      ]);

      const [psnrData, corrData] = await Promise.all([psnrRes.json(), corrRes.json()]);

      if (psnrData.success && corrData.success) {
        setMetrics({ psnr: psnrData.psnr, correlation: corrData.correlation });
      } else {
        setError("Failed to calculate metrics");
      }
    } catch (err) {
      setError(`Failed to calculate metrics: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <header className="bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Image Processing Toolkit
          </h1>
          
          
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Control Panel */}
          <div className="lg:col-span-1 space-y-6">
            {/* Image Upload */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Upload Image
              </h2>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                ref={fileInputRef}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full py-3 px-4 border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg text-gray-600 dark:text-gray-300 hover:border-blue-500 hover:text-blue-500 transition-colors"
              >
                {originalImage ? "Change Image" : "Click to Upload"}
              </button>
            </div>

            {/* Operations */}
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Operations
              </h2>
              <div className="space-y-4">
                {categories.map(category => (
                  <div key={category}>
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                      {category}
                    </h3>
                    <div className="grid grid-cols-2 gap-2">
                      {(Object.entries(operations) as [ProcessingOperation, OperationConfig][])
                        .filter(([, config]) => config.category === category)
                        .map(([key, config]) => (
                          <button
                            key={key}
                            onClick={() => handleOperationChange(key)}
                            className={`py-2 px-3 text-sm rounded-md transition-colors ${
                              selectedOperation === key
                                ? "bg-blue-600 text-white"
                                : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                            }`}
                          >
                            {config.name}
                          </button>
                        ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Parameters */}
            {operations[selectedOperation].params && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Parameters
                </h2>
                <div className="space-y-4">
                  {operations[selectedOperation].params?.map(param => (
                    <div key={param.name}>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                        {param.name}: {params[param.name] ?? param.default}
                      </label>
                      <input
                        type="range"
                        min={param.min}
                        max={param.max}
                        step={param.step}
                        value={params[param.name] ?? param.default}
                        onChange={(e) => setParams(prev => ({ ...prev, [param.name]: parseFloat(e.target.value) }))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Process Button */}
            <button
              onClick={processImage}
              disabled={!originalImage || loading}
              className="w-full py-3 px-4 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? "Processing..." : "Process Image"}
            </button>

            {/* Calculate Metrics Button */}
            {processedImage && selectedOperation !== "attack-scenario" && (
              <button
                onClick={calculateMetrics}
                disabled={loading}
                className="w-full py-3 px-4 bg-green-600 text-white font-semibold rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                Calculate PSNR & Correlation
              </button>
            )}

            {/* Metrics Display */}
            {metrics && (
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Quality Metrics
                </h2>
                <div className="space-y-2">
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    <span className="font-medium">PSNR:</span> {metrics.psnr === Infinity ? "∞ (identical)" : `${metrics.psnr?.toFixed(2)} dB`}
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    <span className="font-medium">Correlation:</span> {metrics.correlation?.toFixed(6)}
                  </p>
                </div>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            )}
          </div>

          {/* Image Display */}
          <div className="lg:col-span-2 space-y-6">
            {attackScenarioResults ? (
              /* Attack Scenario Results */
              <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                  Attack-Decryption Scenario Results
                </h2>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">1. Original</h3>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden aspect-square flex items-center justify-center">
                      <img src={attackScenarioResults.original} alt="Original" className="max-w-full max-h-full object-contain" />
                    </div>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">2. Encrypted</h3>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden aspect-square flex items-center justify-center">
                      <img src={attackScenarioResults.encrypted} alt="Encrypted" className="max-w-full max-h-full object-contain" />
                    </div>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">3. Attacked (Noise Added)</h3>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden aspect-square flex items-center justify-center">
                      <img src={attackScenarioResults.attacked} alt="Attacked" className="max-w-full max-h-full object-contain" />
                    </div>
                  </div>
                  <div>
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">4. Decrypted</h3>
                    <div className="bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden aspect-square flex items-center justify-center">
                      <img src={attackScenarioResults.decrypted} alt="Decrypted" className="max-w-full max-h-full object-contain" />
                    </div>
                  </div>
                </div>
                <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                  <h3 className="font-medium text-gray-900 dark:text-white mb-2">Quality Assessment</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-300">
                    PSNR: {attackScenarioResults.psnr.toFixed(2)} dB | Correlation: {attackScenarioResults.correlation.toFixed(6)}
                  </p>
                </div>
              </div>
            ) : (
              /* Normal Processing Results */
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                  <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Original Image
                  </h2>
                  <div className="bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden aspect-square flex items-center justify-center">
                    {originalImage ? (
                      <img src={originalImage} alt="Original" className="max-w-full max-h-full object-contain" />
                    ) : (
                      <p className="text-gray-400">No image uploaded</p>
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
                  <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                    Processed Image
                  </h2>
                  <div className="bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden aspect-square flex items-center justify-center">
                    {processedImage ? (
                      <img src={processedImage} alt="Processed" className="max-w-full max-h-full object-contain" />
                    ) : (
                      <p className="text-gray-400">No processed image</p>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Info Card */}
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">
                About {operations[selectedOperation].name}
              </h3>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                {getOperationDescription(selectedOperation)}
              </p>
            </div>
          </div>
        </div>
      </main>

      <footer className="bg-white dark:bg-gray-800 mt-12 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500 dark:text-gray-400">
          AIU Image Toolkit - Based on CSE281 Image Processing and Cryptography Lectures
        </div>
      </footer>
    </div>
  );
}

function getOperationDescription(op: ProcessingOperation): string {
  const descriptions: Record<ProcessingOperation, string> = {
    grayscale: "Converts an RGB image to grayscale using the luminosity method: 0.2989*R + 0.5870*G + 0.1140*B (Lectures 4 & 5)",
    negative: "Applies the image negative operation: S = 255 - r. Dark pixels become light and vice versa (Lecture 4)",
    log: "Applies logarithmic transformation: s = c * ln(1 + r). Expands dark pixel values and compresses bright ones (Lectures 4 & 5)",
    gamma: "Applies power-law (gamma) transformation: s = c * r^γ. Gamma < 1 brightens, gamma > 1 darkens (Lectures 4 & 5)",
    histogram: "Applies histogram equalization to improve contrast by redistributing pixel intensities (Lecture 5)",
    binary: "Converts image to binary (black/white) based on threshold value (Lecture 4)",
    "salt-pepper": "Adds Salt & Pepper noise by randomly setting pixels to 0 or 255 (Lecture 6)",
    gaussian: "Applies Gaussian smoothing filter (low-pass) for noise reduction and blurring (Lecture 6)",
    median: "Applies median filter - highly effective for removing Salt & Pepper noise (Lecture 6)",
    spectrum: "Computes the DFT magnitude spectrum for frequency domain visualization (Lecture 8)",
    lpf: "Applies Ideal Low-Pass Filter to remove high frequencies (smoothing/blurring) (Lecture 8)",
    hpf: "Applies Ideal High-Pass Filter to remove low frequencies (edge enhancement) (Lecture 8)",
    notch: "Applies Notch filter to eliminate specific periodic noise frequencies (Lecture 8)",
    encrypt: "Encrypts the image using XOR operation with a secret key (Lecture 8)",
    decrypt: "Decrypts the image using XOR operation with the same key used for encryption (Lecture 8)",
    "attack-scenario": "Demonstrates the complete attack-decryption scenario: Encrypt → Add Noise → Decrypt → Measure Quality (Lecture 8)",
  };
  return descriptions[op];
}
