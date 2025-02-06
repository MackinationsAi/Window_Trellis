import gradio as gr
import numpy as np
import imageio
import shutil
import torch
import json
import os

from typing import *
from PIL import Image
from datetime import datetime
from utils.frontend import load_css
from easydict import EasyDict as edict

from gradio_litmodel3d import LitModel3D

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.representations import Gaussian, MeshExtractResult


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_output_path(params: dict) -> Tuple[str, str]:
    timestamp = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join(OUTPUTS_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    unique_id = datetime.now().strftime('%H-%M-%S')
    param_parts = [unique_id]
    
    if 'seed' in params:
        param_parts.append(f"seed{params['seed']}")
    if all(key in params for key in ['ss_guidance_strength', 'ss_sampling_steps']):
        param_parts.append(f"ss{params['ss_guidance_strength']}_{params['ss_sampling_steps']}")
    if all(key in params for key in ['slat_guidance_strength', 'slat_sampling_steps']):
        param_parts.append(f"slat{params['slat_guidance_strength']}_{params['slat_sampling_steps']}")

    if params.get('is_multiimage'):
        param_parts.append(f"multi_{params.get('multiimage_algo', 'unknown')}")
    if 'mesh_simplify' in params:
        param_parts.append(f"simplify{params['mesh_simplify']}")
    if 'texture_size' in params:
        param_parts.append(f"tex{params['texture_size']}")
        
    param_str = '_'.join(param_parts)
    return output_dir, param_str


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    processed_image = pipeline.preprocess_image(image)
    return processed_image


def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    images = [image[0] for image in images]
    processed_images = [pipeline.preprocess_image(image) for image in images]
    return processed_images


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }
    
    
def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')
    
    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )
    
    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    req: gr.Request,
) -> Tuple[dict, str]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))

    gen_params = {
        'seed': seed,
        'ss_guidance_strength': ss_guidance_strength,
        'ss_sampling_steps': ss_sampling_steps,
        'slat_guidance_strength': slat_guidance_strength,
        'slat_sampling_steps': slat_sampling_steps,
        'is_multiimage': is_multiimage,
    }
    if is_multiimage:
        gen_params['multiimage_algo'] = multiimage_algo
    
    if not is_multiimage:
        outputs = pipeline.run(
            image,
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    else:
        outputs = pipeline.run_multi_image(
            [image[0] for image in multiimages],
            seed=seed,
            formats=["gaussian", "mesh"],
            preprocess_image=False,
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
            mode=multiimage_algo,
        )

    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]

    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)

    output_dir, filename = get_output_path(gen_params)
    output_video_path = os.path.join(output_dir, f"{filename}.mp4")
    imageio.mimsave(output_video_path, video, fps=15)

    state = {
        'gaussian': {
            **outputs['gaussian'][0].init_params,
            '_xyz': outputs['gaussian'][0]._xyz.cpu().numpy(),
            '_features_dc': outputs['gaussian'][0]._features_dc.cpu().numpy(),
            '_scaling': outputs['gaussian'][0]._scaling.cpu().numpy(),
            '_rotation': outputs['gaussian'][0]._rotation.cpu().numpy(),
            '_opacity': outputs['gaussian'][0]._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': outputs['mesh'][0].vertices.cpu().numpy(),
            'faces': outputs['mesh'][0].faces.cpu().numpy(),
        },
        'params': gen_params,
    }
    
    torch.cuda.empty_cache()
    return state, video_path


def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    req: gr.Request,
) -> Tuple[str, str]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)

    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)

    params = {
        **(state.get('params', {})),
        'mesh_simplify': mesh_simplify,
        'texture_size': texture_size
    }

    output_dir, filename = get_output_path(params)
    output_glb_path = os.path.join(output_dir, f"{filename}.glb")
    glb.export(output_glb_path)
    
    torch.cuda.empty_cache()
    return glb_path, glb_path


def extract_gaussian(
    state: dict,
    req: gr.Request
) -> Tuple[str, str]:
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)

    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)

    params = state.get('params', {})

    output_dir, filename = get_output_path(params)
    output_gaussian_path = os.path.join(output_dir, f"{filename}.ply")
    gs.save_ply(output_gaussian_path)
    
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


def prepare_multi_example() -> List[Image.Image]:
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        _images = []
        for i in range(1, 4):
            img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            _images.append(np.array(img))
        images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images


def split_image(image: Image.Image) -> List[Image.Image]:
    image = np.array(image)
    alpha = image[..., 3]
    alpha = np.any(alpha>0, axis=0)
    start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
    end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image[:, s:e+1]))
    return [preprocess_image(image) for image in images]


css = load_css()
utils_folder = "utils"

themes_path = os.path.join(utils_folder, 'themes.json')
config_path = os.path.join(utils_folder, 'config.json')
with open(themes_path, 'r') as f:
    themes = json.load(f)['themes']

if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        default_theme = config.get('theme', 'MackinationsAi/dark_evo')
else:
    default_theme = 'MackinationsAi/dark_evo'

def save_theme_to_config(theme):
    with open(config_path, 'w') as f:
        json.dump({'theme': theme}, f)
    return "Theme saved! Please restart the app to apply the new theme."


def ui(theme):
    with gr.Blocks(theme=theme, css=css, delete_cache=(600, 600)) as TRELLIS:        
        with gr.TabItem(label='Window.Trellis'):
            with gr.Row():
                with gr.Column():
                    with gr.Tabs() as input_tabs:
                        with gr.Tab(label="Single Image", id=0) as single_image_input_tab:
                            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=350)
                        with gr.Tab(label="Multiple Images", id=1) as multiimage_input_tab:
                            multiimage_prompt = gr.Gallery(label="Image Prompt", format="png", type="pil", height=350, columns=3, elem_classes="gallery-scroll")
                    
                    with gr.Accordion(label="Generation Settings", open=True):
                        with gr.Row():
                            with gr.Column():    
                                seed = gr.Slider(0, MAX_SEED, label="Seed", value=0, step=1)
                                randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                            with gr.Column():
                                multiimage_algo = gr.Radio(["stochastic", "multidiffusion"], label="Multi-image Algorithm", value="stochastic", scale=0)
                        gr.Markdown("Stage 1: Sparse Structure Generation")
                        with gr.Row():
                            ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=25, step=1)
                        gr.Markdown("Stage 2: Structured Latent Generation")
                        with gr.Row():
                            slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                            slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=25, step=1)

                    generate_btn = gr.Button("Generate", elem_id="inverse_generate_btn")
                    
                    with gr.Accordion(label="GLB Extraction Settings", open=True):
                        with gr.Row():
                            mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.94, step=0.01)
                            texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)
                    
                    with gr.Row():
                        extract_glb_btn = gr.Button("Extract GLB", interactive=False, elem_id="inverse_generate_btn")
                        extract_gs_btn = gr.Button("Extract Gaussian", interactive=False, elem_id="inverse_generate_btn")

                with gr.Column():
                    video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=402)
                    model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=600)
                    
                    with gr.Row():
                        download_glb = gr.DownloadButton(label="Download GLB", interactive=False, elem_id="generate_btn")
                        download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False, elem_id="generate_btn")  
            
            is_multiimage = gr.State(False)
            output_buf = gr.State()

            with gr.Row() as single_image_example:
                examples = gr.Examples(
                    examples=[
                        f'assets/example_image/{image}'
                        for image in os.listdir("assets/example_image")
                    ],
                    inputs=[image_prompt],
                    fn=preprocess_image,
                    outputs=[image_prompt],
                    run_on_click=True,
                    examples_per_page=64,
                )
            with gr.Row(visible=False) as multiimage_example:
                examples_multi = gr.Examples(
                    examples=prepare_multi_example(),
                    inputs=[image_prompt],
                    fn=split_image,
                    outputs=[multiimage_prompt],
                    run_on_click=True,
                    examples_per_page=8,
                )

            TRELLIS.load(start_session)
            TRELLIS.unload(end_session)
            
            single_image_input_tab.select(
                lambda: tuple([False, gr.Row.update(visible=True), gr.Row.update(visible=False)]),
                outputs=[is_multiimage, single_image_example, multiimage_example]
            )
            multiimage_input_tab.select(
                lambda: tuple([True, gr.Row.update(visible=False), gr.Row.update(visible=True)]),
                outputs=[is_multiimage, single_image_example, multiimage_example]
            )
            
            image_prompt.upload(
                preprocess_image,
                inputs=[image_prompt],
                outputs=[image_prompt],
            )
            multiimage_prompt.upload(
                preprocess_images,
                inputs=[multiimage_prompt],
                outputs=[multiimage_prompt],
            )

            generate_btn.click(
                get_seed,
                inputs=[randomize_seed, seed],
                outputs=[seed],
            ).then(
                image_to_3d,
                inputs=[image_prompt, multiimage_prompt, is_multiimage, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, multiimage_algo],
                outputs=[output_buf, video_output],
            ).then(
                lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
                outputs=[extract_glb_btn, extract_gs_btn],
            )

            video_output.clear(
                lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
                outputs=[extract_glb_btn, extract_gs_btn],
            )

            extract_glb_btn.click(
                extract_glb,
                inputs=[output_buf, mesh_simplify, texture_size],
                outputs=[model_output, download_glb],
            ).then(
                lambda: gr.Button(interactive=True),
                outputs=[download_glb],
            )
            
            extract_gs_btn.click(
                extract_gaussian,
                inputs=[output_buf],
                outputs=[model_output, download_gs],
            ).then(
                lambda: gr.Button(interactive=True),
                outputs=[download_gs],
            )

            model_output.clear(
                lambda: gr.Button(interactive=False),
                outputs=[download_glb],
            )
        
        with gr.TabItem('UI.Settings'):
            with gr.Row():
                with gr.Column(scale=2):
                    theme_dropdown = gr.Dropdown(label="Select Theme", choices=themes, value=themes[0])
                with gr.Column(scale=0.5):
                    apply_theme_button = gr.Button(value="Save New Theme", interactive=True, elem_id="theme_btn")
                    restart_message = gr.HTML(value="Click save new theme & then restart app to apply it.", visible=True)
            permanent_image = LitModel3D(value="utils/static/trellis_T.glb", exposure=10.0, show_label=False, elem_id="permanent_image", interactive=False)
            apply_theme_button.click(
                fn=save_theme_to_config,
                inputs=[theme_dropdown],
                outputs=[restart_message])
        
    return TRELLIS

TRELLIS = ui(default_theme)
if __name__ == "__main__":
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    TRELLIS.launch(server_name='127.0.0.1', share=False)
else:
    print("Failed to create the TRELLIS Gradio interface.")
