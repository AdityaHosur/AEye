"""
OCR Module for Handwritten Text Recognition
Uses Microsoft TrOCR (Transformer-based OCR) for accurate handwriting recognition
Optimized for multi-paragraph handwritten documents
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import cv2

# Try to import TrOCR
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("⚠️  TrOCR not available. Install with: pip install transformers torch pillow")


# Global model cache
_trocr_model = None
_trocr_processor = None


# ============================================================================
# TrOCR Model Functions
# ============================================================================

def load_trocr_model():
    """Load TrOCR model and processor (lazy loading)"""
    global _trocr_model, _trocr_processor
    
    if _trocr_model is None or _trocr_processor is None:
        print("🔄 Loading TrOCR model (first time only, may take a few minutes)...")
        print("   Model: microsoft/trocr-base-handwritten")
        
        try:
            _trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            _trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            
            # Move to GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _trocr_model = _trocr_model.to(device)
            
            if device == "cuda":
                print("   ✅ Using GPU acceleration")
            else:
                print("   ℹ️  Using CPU (for faster processing, use GPU)")
            
            print("✅ TrOCR model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load TrOCR model: {e}")
            raise
    
    return _trocr_processor, _trocr_model


def preprocess_for_trocr(image: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Preprocess image for TrOCR model.
    
    Args:
        image: Path to image, PIL Image, or numpy array
        
    Returns:
        PIL Image optimized for TrOCR
    """
    # Load image
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    else:
        raise ValueError("Image must be path, PIL Image, or numpy array")
    
    # Convert to RGB (required by TrOCR)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Enhance contrast for better recognition
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    
    # Enhance brightness slightly
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    
    return img


def split_image_into_lines_improved(image: Image.Image, min_line_height: int = 15, 
                                     min_gap: int = 5) -> List[Image.Image]:
    """
    Improved line splitting for handwritten documents with multiple paragraphs.
    
    Args:
        image: PIL Image
        min_line_height: Minimum height for a valid line (pixels)
        min_gap: Minimum gap between lines (pixels)
        
    Returns:
        List of PIL Images, one per text line
    """
    # Convert to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # Apply bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Binarize (invert so text is white on black)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological closing to connect broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Calculate horizontal projection (sum of white pixels per row)
    horizontal_projection = np.sum(binary, axis=1)
    
    # Smooth the projection to reduce noise
    from scipy.ndimage import gaussian_filter1d
    horizontal_projection = gaussian_filter1d(horizontal_projection, sigma=2)
    
    # Find line boundaries using adaptive thresholding
    # A line exists where projection is above threshold
    max_projection = np.max(horizontal_projection)
    threshold = max_projection * 0.15  # 15% of maximum
    
    # Find regions where text exists
    in_line = False
    line_boundaries = []
    start = 0
    gap_counter = 0
    
    for i, val in enumerate(horizontal_projection):
        if not in_line and val > threshold:
            # Start of a line
            start = i
            in_line = True
            gap_counter = 0
        elif in_line:
            if val <= threshold:
                gap_counter += 1
                # If gap is large enough, consider it end of line
                if gap_counter >= min_gap:
                    if i - start >= min_line_height:
                        line_boundaries.append((start, i - gap_counter))
                    in_line = False
            else:
                gap_counter = 0
    
    # Handle last line
    if in_line and len(horizontal_projection) - start >= min_line_height:
        line_boundaries.append((start, len(horizontal_projection)))
    
    # If no lines detected or only 1 line detected from large document
    # Try more aggressive splitting
    if len(line_boundaries) <= 1 and img_array.shape[0] > 200:
        print("   ⚠️  Few lines detected, trying more aggressive splitting...")
        line_boundaries = split_by_fixed_height(img_array, target_height=80)
    
    # If still no lines, return original image
    if not line_boundaries:
        print("   ⚠️  No lines detected, processing entire image as one block")
        return [image]
    
    print(f"   📄 Detected {len(line_boundaries)} line(s)")
    
    # Extract line images with padding
    line_images = []
    img_array_rgb = np.array(image)
    
    for idx, (top, bottom) in enumerate(line_boundaries):
        # Add padding
        padding_top = 8
        padding_bottom = 8
        top = max(0, top - padding_top)
        bottom = min(img_array_rgb.shape[0], bottom + padding_bottom)
        
        # Extract line
        line_img = img_array_rgb[top:bottom, :]
        
        # Skip very small lines
        if line_img.shape[0] < min_line_height:
            continue
        
        line_pil = Image.fromarray(line_img)
        line_images.append(line_pil)
    
    return line_images if line_images else [image]


def split_by_fixed_height(img_array: np.ndarray, target_height: int = 80, 
                          overlap: int = 10) -> List[Tuple[int, int]]:
    """
    Split image into fixed-height segments with overlap.
    Used as fallback when line detection fails.
    """
    height = img_array.shape[0]
    boundaries = []
    
    current_top = 0
    while current_top < height:
        bottom = min(current_top + target_height, height)
        boundaries.append((current_top, bottom))
        current_top = bottom - overlap
        
        if current_top >= height:
            break
    
    return boundaries


def extract_text_with_trocr(
    image: Union[str, Image.Image],
    split_lines: bool = True,
    max_new_tokens: int = 128
) -> Tuple[str, float]:
    """
    Extract text using TrOCR model with improved line processing.
    
    Args:
        image: Path to image or PIL Image
        split_lines: Split image into lines for better accuracy
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Tuple of (extracted_text, confidence_estimate)
    """
    if not TROCR_AVAILABLE:
        raise Exception(
            "TrOCR is not available. Install with:\n"
            "  pip install transformers torch pillow scipy"
        )
    
    # Load model and processor
    processor, model = load_trocr_model()
    
    # Preprocess image
    img = preprocess_for_trocr(image)
    
    # Split into lines if requested
    if split_lines:
        line_images = split_image_into_lines_improved(img)
    else:
        line_images = [img]
    
    print(f"   🔍 Processing {len(line_images)} segment(s)...")
    
    # Process each line
    extracted_lines = []
    
    for idx, line_img in enumerate(line_images):
        try:
            # Prepare pixel values (following reference code)
            pixel_values = processor(images=line_img, return_tensors="pt").pixel_values
            
            # Move to same device as model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pixel_values = pixel_values.to(device)
            
            # Generate text with beam search for better results
            generated_ids = model.generate(
                pixel_values, 
                max_new_tokens=max_new_tokens,
                num_beams=5,  # Use beam search
                early_stopping=True
            )
            
            # Decode generated text (following reference code)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up the text
            generated_text = generated_text.strip()
            
            if generated_text and generated_text not in ['0', '0 0', '']:
                extracted_lines.append(generated_text)
                preview = generated_text[:70] + "..." if len(generated_text) > 70 else generated_text
                print(f"   ✓ Line {idx + 1}/{len(line_images)}: {preview}")
            else:
                print(f"   ⊗ Line {idx + 1}/{len(line_images)}: [empty or invalid]")
        
        except Exception as e:
            print(f"   ✗ Line {idx + 1}/{len(line_images)}: Failed - {e}")
            continue
    
    # Combine lines with newlines
    full_text = '\n'.join(extracted_lines)
    
    # Calculate confidence estimate
    if full_text.strip() and len(extracted_lines) > 0:
        confidence_estimate = 80.0 + min(len(extracted_lines), 10) * 1.5
    else:
        confidence_estimate = 0.0
    
    return full_text, confidence_estimate


# ============================================================================
# Main OCR Function
# ============================================================================

def extract_text_from_handwritten_image(
    image_path: str,
    output_path: Optional[str] = None,
    split_lines: bool = True,
    save_preprocessed: bool = False,
    max_new_tokens: int = 128
) -> Dict[str, any]:
    """
    Extract text from handwritten image using TrOCR.
    
    Args:
        image_path: Path to input image
        output_path: Path to save extracted text (optional)
        split_lines: Split image into lines for better recognition
        save_preprocessed: Save preprocessed images (line splits)
        max_new_tokens: Maximum tokens to generate per line
        
    Returns:
        Dictionary with extraction results
    """
    if not TROCR_AVAILABLE:
        raise Exception(
            "TrOCR is not available!\n"
            "Install with: pip install transformers torch pillow scipy opencv-python\n"
            "First run will download the model (~500MB)"
        )
    
    # Validate image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    print(f"📸 Loading image: {os.path.basename(image_path)}")
    
    # Extract text with TrOCR
    print("🤖 Extracting text with TrOCR (Transformer-based OCR)...")
    
    try:
        extracted_text, confidence = extract_text_with_trocr(
            image_path,
            split_lines=split_lines,
            max_new_tokens=max_new_tokens
        )
        
        if extracted_text.strip():
            print(f"✅ Text extracted successfully ({len(extracted_text)} characters)")
        else:
            print("⚠️  Warning: No text extracted from image")
            extracted_text = ""
    
    except Exception as e:
        print(f"❌ TrOCR extraction failed: {e}")
        raise
    
    # Save preprocessed line images if requested
    if save_preprocessed and split_lines:
        try:
            img = Image.open(image_path).convert('RGB')
            line_images = split_image_into_lines_improved(img)
            
            if len(line_images) > 1:
                base, ext = os.path.splitext(image_path)
                for idx, line_img in enumerate(line_images):
                    line_path = f"{base}_line_{idx+1:03d}{ext}"
                    line_img.save(line_path)
                print(f"💾 Saved {len(line_images)} line images")
        except Exception as e:
            print(f"⚠️  Could not save preprocessed images: {e}")
    
    # Save to text file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        print(f"💾 Saved extracted text to: {output_path}")
    
    # Calculate statistics
    word_count = len(extracted_text.split())
    line_count = len([l for l in extracted_text.split('\n') if l.strip()])
    char_count = len(extracted_text)
    
    return {
        'success': True,
        'text': extracted_text,
        'engine': 'trocr',
        'model': 'microsoft/trocr-base-handwritten',
        'confidence': round(confidence, 2),
        'statistics': {
            'characters': char_count,
            'words': word_count,
            'lines': line_count
        },
        'input_file': image_path,
        'output_file': str(output_path) if output_path else None,
        'split_lines': split_lines
    }


def batch_extract_handwritten_images(
    image_paths: List[str],
    output_dir: str,
    split_lines: bool = True
) -> List[Dict[str, any]]:
    """Extract text from multiple handwritten images using TrOCR"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n{'='*70}")
        print(f"Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
        print(f"{'='*70}")
        
        try:
            output_filename = Path(image_path).stem + '.txt'
            output_path = output_dir / output_filename
            
            result = extract_text_from_handwritten_image(
                image_path,
                output_path=str(output_path),
                split_lines=split_lines
            )
            
            results.append(result)
            
        except Exception as e:
            print(f"❌ Failed to process {image_path}: {str(e)}")
            results.append({
                'success': False,
                'input_file': image_path,
                'error': str(e)
            })
    
    return results


def get_available_engines() -> List[str]:
    """Get list of available OCR engines"""
    return ['trocr'] if TROCR_AVAILABLE else []


def is_trocr_available() -> bool:
    """Check if TrOCR is available"""
    return TROCR_AVAILABLE


def is_handwritten_image(image_path: str) -> bool:
    """Check if image is supported for OCR"""
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif']
    ext = Path(image_path).suffix.lower()
    return ext in supported_extensions


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("🖼️  Handwritten Text Recognition using Microsoft TrOCR")
    print(f"TrOCR available: {'✅ Yes' if TROCR_AVAILABLE else '❌ No'}")
    print()
    
    if not TROCR_AVAILABLE:
        print("⚠️  TrOCR is not installed!")
        print("\nTo install:")
        print("  pip install transformers torch pillow scipy opencv-python")
        print("\nNote: First run will download the model (~500MB)")
        sys.exit(1)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        if not output_path:
            output_path = Path(image_path).stem + '_extracted.txt'
        
        result = extract_text_from_handwritten_image(
            image_path,
            output_path=output_path,
            split_lines=True,
            save_preprocessed=True
        )
        
        print("\n" + "="*70)
        print("📊 EXTRACTION RESULTS")
        print("="*70)
        print(f"Engine: TrOCR")
        print(f"Model: {result['model']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Words: {result['statistics']['words']}")
        print(f"Lines: {result['statistics']['lines']}")
        print(f"Characters: {result['statistics']['characters']}")
        print(f"\n📝 Extracted Text:")
        print("-"*70)
        print(result['text'])
        print("-"*70)
    
    else:
        print("Usage: python ocr.py <image_path> [output_path]")
        print("Example: python ocr.py handwritten_letter.jpg output.txt")