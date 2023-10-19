# RealtimeGen for AI-painting based on Stable Diffuison

Recent advances in AIGC technology enable the rapid generation of high-quality images and draw the attention from painters. However, integrating the end-to-end image generation principle with the step-by-step human painting process remains challenging. Existing research has revealed the limitations of fully automated technology for professionals. We designed **RealtimeGen**, an interventable interactive image generation tool. In fact, it can be used for Adobe Photoshop. We notice the intrinsic step-by-step principle behind diffusion models, an advanced generation technology. By exposing the full generation process, RealtimeGen allows painters to integrate both AI generation and human painting process.


Here we demonstrate how to do preview images of a Stable Diffusion's intermediate stages using a fast approximation to visualize the low-resolution (64px) latent state.
支持SDXL、SD2、SDV1.5
* app.py is a Gradio application that yields preview images from a generator function while the pipeline is in progress. The UI is directly derived from Stability AI's Stable Diffusion Demo.
progress_ipywidgets_demo.ipynb demonstrates using the same pipeline to update Jupyter widgets in a notebook.
* preview_decoder.py has the fast latent-to-RGB decoder function.
* item.py 
back_utils
client.py
generator_pipeline.py provides a DiffusionPipeline with a generate() method to yield the latent data at each step. It is nearly a strict refactoring of the StableDiffusionPipeline in 🧨diffusers 0.19.
```
