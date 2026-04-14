# 📷 Photo Organizer

An AI-powered tool that automatically groups and organizes photos by identifying the people in them. It uses state-of-the-art face detection and recognition to cluster similar faces and allows users to interactively name and organize their image libraries.

## ✨ Features

- **AI Face Recognition**: Uses facial encodings to distinguish between different individuals across thousands of photos.
- **Intelligent Clustering**: Implements the DBSCAN algorithm to group similar faces into clusters without requiring a predefined number of people.
- **Incremental Processing**: A smart SQLite-based caching system ensures that only new or modified photos are processed, drastically reducing execution time for large libraries.
- **Interactive Organization**: A CLI-driven workflow that lets you name discovered clusters and automatically move photos into named folders.
- **High Performance**: 
  - **Parallel Execution**: Utilizes `ProcessPoolExecutor` to leverage multi-core CPUs.
  - **Flexible Detection**: Supports both standard and high-accuracy CNN-based face detectors.
  - **Optimized Loading**: Optional image downscaling to accelerate the detection phase.
- **Broad Format Support**: Full support for standard image formats and HEIC files (via `pillow-heif`).

## 🏗️ Architecture

The project is designed as a modular pipeline:

`main.py` $\rightarrow$ `loader.py` $\rightarrow$ `detector.py` $\rightarrow$ `encoder.py` $\rightarrow$ `clustering.py` $\rightarrow$ `organizer.py`

- **Loading**: Traverses directories and handles image loading.
- **Detection**: Locates faces in images using either HOG or CNN methods.
- **Encoding**: Converts faces into 128-dimensional embeddings.
- **Caching**: Stores embeddings in `cache/cache.db` to avoid redundant computation.
- **Clustering**: Groups embeddings based on spatial proximity.
- **Organization**: Moves files into the `output/` directory based on user-provided names.

## 🚀 Getting Started

### Prerequisites
- Python 3.14
- Required libraries: `face-recognition`, `opencv-python`, `scikit-learn`, `dlib`, `pillow-heif`, `numpy`.

### Installation
Due to the complexity of installing `dlib` on Windows, a pre-compiled wheel is provided in the `/dlib` directory.

```bash
pip install -r requirements.txt

# If dlib fails, install from the local wheel:
pip install dlib/dlib-xxx.whl
```

### Usage
Run the organizer by pointing it to your photo directory:

```bash
py main.py "path/to/photos_directory" [options]
```

#### Available Options:
| Option | Description |
| :--- | :--- |
| `-v, --verbose` | Enable detailed logging. |
| `--force-recompute` | Clear the cache and re-process all images. |
| `--use-cnn` | Use the CNN-based detector (Higher accuracy, slower). |
| `--downscale` | Downscale images before detection to increase speed. |
| `--parallel` | Enable multi-core parallel processing. |

## 📈 Development Evolution

The project evolved through several distinct phases to reach its current state:

1. **Foundational Pipeline**: Establishing the basic flow of scanning folders, detecting faces, and generating initial encodings.
2. **Efficiency & Caching**: Introduction of the SQLite layer to support incremental scans and avoid processing the same image twice.
3. **Clustering & Refinement**: Implementation of DBSCAN and hyper-parameter tuning to ensure high-quality face grouping.
4. **Interactive UX**: Adding the ability for users to interactively name clusters and the automated file-system organization logic.
5. **Performance Scaling**: Transitioning from `ThreadPoolExecutor` to `ProcessPoolExecutor` to bypass the Python GIL and fully utilize multi-core hardware.
6. **Stability & Polishing**: Final refinements including path normalization, crash fixes for empty clusters, and image downscaling for performance.

## 📁 Project Structure
- `main.py`: Pipeline orchestrator and CLI entry point.
- `src/`: Core logic modules (Detector, Encoder, Cache, etc.).
- `cache/`: Persistent storage for facial embeddings.
- `output/`: Destination for organized photos.
- `dlib/`: Pre-compiled binaries for Windows installation.
