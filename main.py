from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from diffusers import DiffusionPipeline
import os
import torch
import io
import base64
import time
from PIL import Image
import json
from pathlib import Path
import threading

# Installation instructions:
# pip install fastapi uvicorn pydantic diffusers transformers accelerate
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# uvicorn main:app --reload

class ImageGenerationRequest(BaseModel):
    """Request schema for image generation"""
    model: Optional[str] = "stabilityai/sdxl-turbo"
    prompt: str
    n: Optional[int] = 1  # Default to 1 image
    size: Optional[str] = "512x512"  # Default to 512x512

class ImageResponse(BaseModel):
    """Response schema for a single generated image"""
    b64_json: str

class ImageGenerationResponse(BaseModel):
    """Response schema for image generation API"""
    created: int
    data: List[ImageResponse]

class ModelInfo(BaseModel):
    """Schema for model information"""
    publisher: str
    family: str
    version: str
    id: str

class ModelsListResponse(BaseModel):
    """Response schema for models listing API"""
    models: List[ModelInfo]

class ReadWriteLock:
    """A simple readers-writer lock."""
    def __init__(self):
        self._readers = 0
        self._read_lock = threading.Lock()
        self._write_lock = threading.Lock()

    def acquire_read(self):
        with self._read_lock:
            self._readers += 1
            if self._readers == 1:
                self._write_lock.acquire()

    def release_read(self):
        with self._read_lock:
            self._readers -= 1
            if self._readers == 0:
                self._write_lock.release()

    def acquire_write(self):
        self._write_lock.acquire()

    def release_write(self):
        self._write_lock.release()

class ModelManager:
    """Class to manage diffusion models"""
    
    def __init__(self, default_model: str = "stabilityai/sdxl-turbo", models_file_path: str = "models_info.json"):
        """Initialize the model manager"""
        self.models_file_path = Path(models_file_path)
        self._available_models = self._load_available_models()
        
        # Verify default model is in available models
        if not any(model.id == default_model for model in self._available_models):
            available_ids = [model.id for model in self._available_models]
            if available_ids:
                default_model = available_ids[0]  # Use first available model
                print(f"Specified default model not available. Using '{default_model}' instead.")
            else:
                raise ValueError("No models available in the models file")
                
        self.current_model_id = default_model
        self._pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Running on {'GPU' if self.device == 'cuda' else 'CPU'}")
        # Initialize a readers-writer lock for concurrency control
        self._rw_lock = ReadWriteLock()
        # Dictionary to cache loaded pipelines
        self._pipelines: Dict[str, DiffusionPipeline] = {}
        # Load default model under write lock
        self._rw_lock.acquire_write()
        try:
            self._load_model(default_model)
        finally:
            self._rw_lock.release_write()
    
    def _load_model(self, model_id: str) -> None:
        """Load a model from its local path"""
        try:
            # Find the model info for the given ID
            model_info = next((model for model in self._available_models if model.id == model_id), None)
            if not model_info:
                raise ValueError(f"Model '{model_id}' not found in available models")

            # Load from local path
            try:
                # First try to load the FP16 variant
                self._pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    variant="fp16",
                    #cache_dir="./models"
                )
                print(f"Loaded FP16 variant of {model_id}")
            except Exception as e:
                print(f"Could not load FP16 variant: {str(e)}")
                print(f"Falling back to standard model...")
                # Fall back to the standard model version
                self._pipe = DiffusionPipeline.from_pretrained(
                    model_id,
                    #cache_dir="./models"
                )
                print(f"Loaded standard variant of {model_id}")
        
            self._pipe = self._pipe.to(self.device)
            self.current_model_id = model_id
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_id}': {str(e)}")
    
    def get_pipeline(self, model_id: str) -> DiffusionPipeline:
        """Get the pipeline for the requested model, loading it if necessary"""
        # Check if the model is in the available models list
        if not any(model.id == model_id for model in self._available_models):
            raise ValueError(f"Model '{model_id}' is not in the list of available models")
            
        if model_id != self.current_model_id:
            # Acquire write lock to switch models exclusively
            self._rw_lock.acquire_write()
            try:
                self._unload_current_model()
                print(f"Loading new model: {model_id}")
                self._load_model(model_id)
            finally:
                self._rw_lock.release_write()
        
        return self._pipe
    
    def _unload_current_model(self) -> None:
        """Unload the current model to free memory"""
        if self._pipe is None:
            return
            
        if self.device == "cuda":
            # Move to CPU first (safer approach)
            self._pipe = self._pipe.to("cpu")
            del self._pipe
            
            # Clear the CUDA cache to ensure memory is freed
            torch.cuda.empty_cache()
            print("Previous model unloaded and GPU cache cleared")
        else:
            del self._pipe
            
        self._pipe = None
        
    def _load_available_models(self) -> List[ModelInfo]:
        """Load the list of available models from the models_info.json file or create if not exists"""
        models_file = Path("models_info.json")
        
        # If the file doesn't exist, create it with default models
        if not models_file.exists():
            default_models = [
                {
                    "publisher": "stabilityai",
                    "family": "sdxl",
                    "version": "turbo",
                    "id": "stabilityai/sdxl-turbo"
                },
            ]
            
            with open(models_file, "w") as f:
                json.dump(default_models, f, indent=2)
        
        # Load models from the file
        with open(models_file, "r") as f:
            models_data = json.load(f)
            
        return [ModelInfo(**model) for model in models_data]
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get the list of available models"""
        return self._available_models


class ImageGenerator:
    """Class to handle image generation"""
    
    def __init__(self, model_manager: ModelManager):
        """Initialize with a model manager"""
        self.model_manager = model_manager
    
    def generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Generate images based on the request"""
        try:
            # Parse size (e.g., "512x512") into width and height
            width, height = map(int, request.size.split("x"))
            
            try:
                # Get the pipeline for the requested model
                pipe = self.model_manager.get_pipeline(request.model)
            except ValueError as e:
                # Convert model validation errors to HTTP exceptions
                raise HTTPException(status_code=400, detail=str(e))
            
            # Generate the requested number of images
            images_data = []
            # Acquire read lock for concurrent generation on the same model
            self.model_manager._rw_lock.acquire_read()
            try:
                for _ in range(request.n):
                    result = pipe(
                        prompt=request.prompt,
                        height=height,
                        width=width,
                        num_inference_steps=25
                    )
                    image = result.images[0]
                    
                    # Convert image to base64 PNG
                    img_base64 = self._image_to_base64(image)
                    images_data.append(ImageResponse(b64_json=img_base64))
            finally:
                self.model_manager._rw_lock.release_read()
            
            # Create and return the response
            return ImageGenerationResponse(
                created=int(time.time()),
                data=images_data
            )
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        """Convert a PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class StableDiffusionAPI:
    """Main API class"""
    
    def __init__(self, models_directory: str = "models", models_file: str = "models_info.json"):
        """Initialize the API"""
        self.app = FastAPI(title="Stable Diffusion API")
        self.models_directory = Path(models_directory)
        
        # Create models directory if it doesn't exist
        if not self.models_directory.exists():
            self.models_directory.mkdir(parents=True)
            print(f"Created models directory at {self.models_directory}")
            
        self.model_manager = ModelManager(models_file_path=models_file)
        self.image_generator = ImageGenerator(self.model_manager)
        self._setup_routes()

        # Add CORS middleware to allow cross-origin requests
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Adjust this to restrict origins in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Set up the API routes"""
        @self.app.post("/v1/images/generations", response_model=ImageGenerationResponse)
        async def generate_image(request: ImageGenerationRequest):
            return self.image_generator.generate(request)
            
        @self.app.get("/v1/models", response_model=ModelsListResponse)
        async def list_models():
            """List all available models"""
            models = self.model_manager.get_available_models()
            return ModelsListResponse(models=models)


# Create and expose the FastAPI application
api = StableDiffusionAPI(models_directory="models", models_file="models_info.json")
app = api.app

# This allows the script to be run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
