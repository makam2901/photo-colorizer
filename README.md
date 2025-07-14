# Photo Colorizer using Diffusion Models

This project demonstrates how to build and train a diffusion model to colorize grayscale images. It uses a conditional U-Net model guided by CLIP text embeddings to generate realistic and context-aware colors.

## Features

* Colorizes grayscale images using a text prompt.
* Utilizes a lightweight, conditional U-Net architecture.
* Leverages the power of CLIP for semantic understanding of image content.
* Includes code for both training the model and running inference on new images.
* Can colorize images from a dataset (CIFAR-10 in this example) or external image files.

## Technologies Used

* Python 3.10
* PyTorch
* Hugging Face Libraries:
    * `diffusers`: For the U-Net model and diffusion schedulers.
    * `datasets`: To load and handle the training data.
    * `transformers`: For the CLIP model and processor.
* scikit-image: For image processing and color space conversions (RGB to LAB).
* NumPy: For numerical operations.
* Matplotlib: For visualizing the results.
* Pillow (PIL): For image manipulation.
* tqdm: For progress bars.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/makam2901/photo-colorizer.git](https://github.com/makam2901/photo-colorizer.git)
    cd photo-colorizer
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install torch torchvision
    pip install diffusers datasets transformers scikit-image numpy matplotlib pillow tqdm ipywidgets
    ```

## Usage

The main logic is contained within the `Project_GenAI_final.ipynb` Jupyter Notebook.

### 1. Training the Model

* Open and run the notebook cells sequentially.
* The initial cells will import libraries, set up the configuration (like image size, batch size, etc.), and prepare the dataset.
* The model is trained on the CIFAR-10 dataset. The notebook preprocesses the images by converting them to the LAB color space and extracts the L (lightness) channel as the input and the AB (color) channels as the target.
* The training loop uses a `DDPMScheduler` and AdamW optimizer to train the U-Net model.

### 2. Running Inference

#### On Dataset Samples

* After training, you can run the inference section of the notebook.
* It provides a function `colorize_image` that takes an L-channel tensor and a text prompt to generate the colorized image.
* A sample from the training dataset is used to demonstrate the colorization, showing the original, grayscale, and colorized images side-by-side.

#### On External Images

* The notebook includes a function `colorize_external_image` to colorize your own images.
* To use it, update the `my_image_path` and `my_prompt` variables in the final cells of the notebook:
    ```python
    my_image_path = "path/to/your/image.jpg"
    my_prompt = "A descriptive prompt for your image."
    generated_image = colorize_external_image(my_image_path, my_prompt, model, config, config.DEVICE)
    ```
* This will display the original, grayscale, and AI-colorized versions of your image.


| *Prompt: 'A high quality photo of a baseball player swinging a bat in an old stadium.'* |

## Future Improvements

* Train on a larger, more diverse dataset of high-resolution images.
* Experiment with more powerful U-Net architectures or different diffusion schedulers.
* Fine-tune the CLIP model or use a larger version for better text-image alignment.
* Deploy the model as a web application using a framework like Flask or Gradio for an interactive user experience.
