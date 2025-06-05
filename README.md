# AI Image Generation API

## Overview

This project provides a production-ready RESTful service for text-to-image generation. Leveraging state-of-the-art Stable Diffusion variants (e.g., SDXL Turbo, Animagine XL) with PyTorch and CUDA support, the API enables rapid, high-quality image creation suitable for web and enterprise usage. Models can be hot-swapped at runtime, and all endpoints follow a structure compatible with OpenAI’s image-generation format.

---

## Key Features

* **Multiple AI Models**
  Support for a variety of Stable Diffusion checkpoints, including SDXL Turbo, Stable Diffusion v1.4, and Animagine XL 3.0.

* **GPU-Accelerated Inference**
  Automatic CUDA utilization for fast image synthesis (falls back to CPU when no GPU is available).

* **Dynamic Model Management**
  Load and switch between models without restarting the server. List available models along with their metadata.

* **RESTful Interface**
  Follows OpenAI’s `/v1/images/generations` format for seamless integration with existing tooling.

* **Dockerized Deployment**
  Ready-to-use Dockerfile with NVIDIA Docker support for containerized GPU inference.

* **Thread Safety**
  Read-write locks around model access guarantee safe concurrent requests.

---

## Supported Models

| Model Identifier                 | Description                            |
| -------------------------------- | -------------------------------------- |
| `stabilityai/sdxl-turbo`         | Fast, high-fidelity SDXL variant       |
| `CompVis/stable-diffusion-v1-4`  | Standard Stable Diffusion v1.4         |
| `Cagliostrolab/animagine-xl-3.0` | Optimized for anime-style image output |

Model entries are defined in `models_info.json`. To add a new model, include its Hugging Face ID and metadata in that file.

---

## Quick Start

### Prerequisites

* Python 3.8 or higher
* NVIDIA GPU with CUDA 12.8+ (for GPU inference)
* Docker 20.10+ (optional, for containerized deployment)

### Local Development

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ai-img-gen
   ```

2. **Install Python requirements**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the API**

   * Base URL: `http://localhost:8000`
   * Interactive Swagger docs: `http://localhost:8000/docs`

### Docker Deployment

1. **Build Docker image**

   ```bash
   docker build -t ai-img-gen .
   ```

2. **Run container with GPU support**

   ```bash
   docker run --gpus all -p 8000:8000 ai-img-gen
   ```

---

## API Reference

### 1. Generate Images

* **Endpoint**

  ```http
  POST /v1/images/generations
  ```

* **Description**
  Produce one or more images based on a text prompt.

* **Request Body** (JSON)

  | Field  | Type    | Description                                  |
  | ------ | ------- | -------------------------------------------- |
  | model  | string  | Hugging Face model ID (e.g., `"sdxl-turbo"`) |
  | prompt | string  | Text description for image generation        |
  | n      | integer | Number of images to generate (default: 1)    |
  | size   | string  | Desired resolution (e.g., `"512x512"`)       |

  ```json
  {
    "model": "stabilityai/sdxl-turbo",
    "prompt": "A serene forest at dawn",
    "n": 2,
    "size": "512x512"
  }
  ```

* **Response** (JSON)

  | Field   | Type    | Description                           |
  | ------- | ------- | ------------------------------------- |
  | created | integer | UNIX timestamp when images were made  |
  | data    | array   | List of objects containing `b64_json` |

  ```json
  {
    "created": 1717545600,
    "data": [
      {
        "b64_json": "<base64-encoded-image-1>"
      },
      {
        "b64_json": "<base64-encoded-image-2>"
      }
    ]
  }
  ```

---

### 2. List Available Models

* **Endpoint**

  ```http
  GET /v1/models
  ```

* **Description**
  Retrieve metadata for all models currently configured.

* **Response** (JSON)

  | Field       | Type   | Description                                             |
  | ----------- | ------ | ------------------------------------------------------- |
  | models      | array  | An array of model metadata objects                      |
  | • publisher | string | Model publisher (e.g., `"stabilityai"`)                 |
  | • family    | string | Model family (e.g., `"sdxl"`)                           |
  | • version   | string | Model version (e.g., `"turbo"`)                         |
  | • id        | string | Full Hugging Face ID (e.g., `"stabilityai/sdxl-turbo"`) |

  ```json
  {
    "models": [
      {
        "publisher": "stabilityai",
        "family": "sdxl",
        "version": "turbo",
        "id": "stabilityai/sdxl-turbo"
      },
      {
        "publisher": "CompVis",
        "family": "stable-diffusion",
        "version": "v1.4",
        "id": "CompVis/stable-diffusion-v1-4"
      }
    ]
  }
  ```

---

## Configuration

### Model Definitions

All models are defined in the `models_info.json` file at the project root. Each entry must include:

```jsonc
{
  "publisher": "your-publisher",
  "family": "model-family",
  "version": "1.0",
  "id": "publisher/model-name"
}
```

To add or update a model:

1. Edit `models_info.json`.
2. Include the full Hugging Face model identifier and relevant metadata.
3. Restart the server to load the new model.
