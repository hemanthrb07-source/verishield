"""
Core preprocessor module for all input types.
Handles resizing, normalization, metadata extraction.
"""
import hashlib
import io
import os
from typing import Optional
from PIL import Image
import numpy as np


class Preprocessor:
    """Universal preprocessor for documents, images, and videos."""
    
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    SUPPORTED_DOC_FORMATS = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}
    
    def __init__(self):
        self.target_size = (224, 224)  # Standard input for most CNNs
        self.max_dimension = 4096
    
    def compute_file_hash(self, file_content: bytes) -> str:
        """Compute SHA-256 hash of file content."""
        return hashlib.sha256(file_content).hexdigest()
    
    def detect_file_type(self, filename: str, content_type: str = "") -> str:
        """Detect file type from extension and content type."""
        ext = os.path.splitext(filename.lower())[1]
        
        if ext in self.SUPPORTED_IMAGE_FORMATS or 'image' in content_type:
            return 'IMAGE'
        elif ext in self.SUPPORTED_VIDEO_FORMATS or 'video' in content_type:
            return 'VIDEO'
        elif ext in self.SUPPORTED_DOC_FORMATS or 'pdf' in content_type:
            return 'DOCUMENT'
        
        # Default to IMAGE for unknown types
        return 'IMAGE'
    
    def preprocess_image(self, file_content: bytes) -> dict:
        """
        Preprocess an image: resize, normalize, extract metadata.
        Returns dict with processed image array and metadata.
        """
        image = Image.open(io.BytesIO(file_content))
        original_size = image.size
        mode = image.mode
        
        # Extract EXIF metadata
        metadata = self._extract_image_metadata(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize for model input
        resized = image.resize(self.target_size, Image.Resampling.LANCZOS)
        image_array = np.array(resized, dtype=np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (image_array - mean) / std
        
        # Transpose to CHW format (C, H, W) for PyTorch
        chw_image = normalized.transpose(2, 0, 1)
        
        return {
            'image_array': chw_image,
            'original_image': np.array(Image.open(io.BytesIO(file_content))),
            'original_size': original_size,
            'mode': mode,
            'metadata': metadata,
            'file_hash': self.compute_file_hash(file_content),
        }
    
    def extract_image_region(self, image_array: np.ndarray, region: dict) -> np.ndarray:
        """Extract a specific region from an image array."""
        x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 100), region.get('h', 100)
        return image_array[y:y+h, x:x+w]
    
    def _extract_image_metadata(self, image: Image.Image) -> dict:
        """Extract metadata from an image."""
        metadata = {
            'format': image.format,
            'mode': image.mode,
            'size': list(image.size),
            'info_keys': list(image.info.keys()) if hasattr(image, 'info') else [],
        }
        
        # Try to extract EXIF data
        try:
            exif_data = image.getexif()
            if exif_data:
                exif_tags = {}
                from PIL.ExifTags import TAGS
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    # Only store serializable values
                    try:
                        json.dumps(value)
                        exif_tags[tag_name] = str(value)
                    except (TypeError, ValueError):
                        exif_tags[tag_name] = str(value)
                metadata['exif'] = exif_tags
                metadata['has_exif'] = True
            else:
                metadata['has_exif'] = False
        except Exception:
            metadata['has_exif'] = False
        
        return metadata
    
    def preprocess_document(self, file_content: bytes, filename: str) -> dict:
        """
        Preprocess a document (PDF or image-based).
        For PDFs, convert first page to image.
        For images, use image preprocessing.
        """
        ext = os.path.splitext(filename.lower())[1]
        
        if ext == '.pdf':
            return self._preprocess_pdf(file_content)
        else:
            result = self.preprocess_image(file_content)
            result['document_type'] = 'image_based'
            return result
    
    def _preprocess_pdf(self, file_content: bytes) -> dict:
        """Preprocess a PDF document."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=file_content, filetype="pdf")
            
            pages = []
            for page_num in range(min(len(doc), 5)):  # Limit to first 5 pages
                page = doc[page_num]
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                image = Image.open(io.BytesIO(img_data))
                image_array = np.array(image)
                
                pages.append({
                    'page_num': page_num,
                    'image': image_array,
                    'text': page.get_text(),
                    'size': list(page.rect),
                })
            
            metadata = {
                'num_pages': len(doc),
                'metadata': dict(doc.metadata) if doc.metadata else {},
            }
            doc.close()
            
            return {
                'pages': pages,
                'metadata': metadata,
                'document_type': 'pdf',
                'file_hash': self.compute_file_hash(file_content),
            }
        except ImportError:
            # Fallback: treat as image
            return self.preprocess_image(file_content)
    
    def preprocess_video(self, file_content: bytes, max_frames: int = 10) -> dict:
        """
        Preprocess a video: extract key frames.
        """
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        frames = []
        try:
            import cv2
            cap = cv2.VideoCapture(tmp_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Sample frames evenly
            frame_indices = np.linspace(0, max(0, total_frames - 1), max_frames, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    pil_image = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
                    frame_array = np.array(pil_image, dtype=np.float32) / 255.0
                    
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    normalized = (frame_array - mean) / std
                    chw_frame = normalized.transpose(2, 0, 1)
                    
                    frames.append({
                        'frame_index': int(idx),
                        'frame_array': chw_frame,
                        'original_frame': frame_rgb,
                    })
            
            cap.release()
            
            metadata = {
                'total_frames': total_frames,
                'fps': fps,
                'duration_seconds': duration,
                'sampled_frames': len(frames),
            }
            
        except ImportError:
            # Fallback: try to treat first bytes as image
            result = self.preprocess_image(file_content[:min(len(file_content), 1024*1024)])
            metadata = {'note': 'cv2 not available, processed as image fallback'}
            frames = [{'frame_array': result['image_array'], 'frame_index': 0}]
        finally:
            os.unlink(tmp_path)
        
        return {
            'frames': frames,
            'metadata': metadata,
            'file_hash': self.compute_file_hash(file_content),
        }
