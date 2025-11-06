## 🧩 Overview

This project implements a **complete end-to-end pipeline** for automated document processing of **semi-structured handwritten time-tracking forms**.  
It combines **document layout analysis (DLA)**, **optical character recognition (OCR)**, **handwritten text recognition (HTR)** and **information extraction** to convert scanned form images into a **structured relational database schema**.  

The system was developed in the context of a Master’s thesis in Computer Science and is designed for extensibility, modularity and transparency of each processing step.

---

## ⚙️ Pipeline Architecture

            +---------------------+
            |   Image Input       |
            +----------+----------+
                       |
                       v
            +---------------------+
            | Image Preprocessing |
            | (denoising, skew,   |
            |  binarization)      |
            +----------+----------+
                       |
                       v
            +---------------------+
            | Layout Analysis     |
            | (LayoutParser + CV) |
            +----------+----------+
                       |
                       v
            +---------------------+
            | OCR / HTR Text      |
            | Recognition         |
            +----------+----------+
                       |
                       v
            +---------------------+
            | Information          |
            | Extraction (forms 1–3)|
            +----------+-----------+
                       |
                       v
            +---------------------+
            | Database Storage     |
            | (PostgreSQL schema)  |
            +---------------------+



---

## 🧠 Core Modules

### 1. `ImagePreprocessor`
**File:** `iamge_preprocessing.py`  
Performs image enhancement, denoising, binarization (Sauvola), skew correction (Hough transform) and morphological reinforcement.  
Output images are stored in `input/preprocessed/`.  

Main steps:  
- Grayscale conversion  
- Median filtering  
- Sauvola binarization  
- Skew correction  
- Morphological cleaning  

---

### 2. `TabularLineEnhancer`
**File:** `tabular_line_enhancer.py`  
Uses OpenCV operations to **detect and enhance table structures** by combining horizontal and vertical morphological filters.  
Provides accurate bounding boxes for tables and their inner cells, which improves the layout analysis.

Key functionalities:  
- Adaptive thresholding  
- Horizontal and vertical line extraction  
- Table contour detection  
- Cell segmentation  
- JSON export of table structure  

---

### 3. `ImageLayoutAnalyzer`
**File:** `image_layout_analysis.py`  
Combines **LayoutParser (Detectron2)** with the `TabularLineEnhancer` to recognize and optimize document layout structures.  
Detected layout elements include *Text*, *Title*, *Table*, and *Cell* blocks.  

Features:  
- Runs Detectron2 model (Faster/Mask R-CNN)  
- Refines table frames via IoU comparison with CV-detected tables  
- Extracts layout statistics (tables, cells, block distributions)  
- Classifies form types via feature comparison  
- Outputs JSON with all bounding boxes  

Output:  
output/lp/layout_parser_form_1_1_<image_name>.json
---

### 4. `ImageTextRecognizer`
**File:** `image_text_recognition.py`  
Performs **text recognition** by combining OCR (DocTR) for printed text and HTR (TrOCR) for handwritten segments.  
It links OCR results to layout cells and classifies each segment as OCR or HTR.  

Pipeline steps:  
1. Layout analysis (via `ImageLayoutAnalyzer`)  
2. OCR via DocTR  
3. Word-to-cell assignment and correction  
4. TrOCR execution on HTR cells  
5. Result visualization and JSON export  

Output files:  
output/ocr/ocr_
output/ocr/ocr_<image_name>_htr.json



---

### 5. `InformationExtractor`
**File:** `information_extraction.py`  
Transforms recognized text segments into **structured information** based on predefined form templates.  
Supports multiple form types (`form_1`, `form_2`, `form_3`) with custom extraction logic.  

Main tasks:  
- Identify metadata and data tables  
- Extract key–value pairs (e.g. “Name”, “Geburtsdatum”)  
- Identify row-based activity entries  
- Align OCR and HTR text spatially  
- Export unified JSON for each form  

Example output:
output/ie/form_3_1A_extracted.json


---

### 6. `DataTransformer`
**File:** `data_trasnformer.py`  
Acts as the **final integration layer** between extracted data and the PostgreSQL database.  
It creates necessary tables and inserts metadata and activity records into a normalized relational schema.  

Database tables include:
- `form_metadata`  
- `taetigkeiten_form1`  
- `taetigkeiten_form2`  
- `taetigkeiten_form3`  

Each entry is linked via a foreign key (`form_id` / `meta_id`) to its metadata.

---

### 7. `main.py`
The main entry script to execute the entire processing pipeline.  

Example:
```python
from src.data_transformation.data_transformer import DataTransformer

def run_processing_document(image_name="form_3_1A.png"):
    data_transformer = DataTransformer(image_name=image_name, rerun_ocr=False)
    # Optional:
    # data_transformer.create_forms_table_if_not_exists()
    # data_transformer.save_data_into_db()

run_processing_document()

```
🗃️ Folder Structure
```
project_root/
│
├── src/
│   ├── preprocessing/
│   │   └── iamge_preprocessing.py
│   ├── layout_analysis/
│   │   ├── image_layout_analysis.py
│   │   └── tabular_line_enhancer.py
│   ├── text_recognition/
│   │   └── image_text_recognition.py
│   ├── information_extraction/
│   │   └── information_extraction.py
│   └── data_transformation/
│       └── data_transformer.py
│
├── input/
│   ├── preprocessed/
│   ├── forms/
│   └── images/
│
├── output/
│   ├── lp/    # LayoutParser results
│   ├── ocr/   # OCR & HTR results
│   └── ie/    # Extracted structured data
│
└── main.py
```

💾 Database Configuration

| Parameter    | Default Value      | Description              |
| ------------ | ------------------ | ------------------------ |
| **dbname**   | `image_processing` | PostgreSQL database name |
| **user**     | `postgres`         | Database user            |
| **password** | `postgres`         | Password                 |
| **host**     | `localhost`        | Database host            |
| **port**     | `5432`             | Port                     |



🧰 Dependencies
```
pip install -r requirements.txt
```
Typical libraries:

```
opencv-python
numpy
matplotlib
layoutparser
pytesseract
doctr
transformers
scipy
psycopg2
scikit-image
```
▶️ Usage

Process a single document
```
python main.py
```

Store recognized data in database
```
data_transformer = DataTransformer(image_name="form_3_1A.png", rerun_ocr=False)
data_transformer.create_forms_table_if_not_exists()
data_transformer.save_data_into_db()
```

Run batch extraction for all images
```
from src.information_extraction.information_extraction import InformationExtractor
extractor = InformationExtractor(input_folder="../../input")
extractor.run_for_all_files()
```

📊 Output Examples
```
Layout visualization: output/lp/layout_parser_<image_name>.png
OCR visualization: output/ocr/ocr_<image_name>.png
HTR segment outputs: output/htr/
Extracted structured data: output/ie/form_x_y_extracted.json
```
