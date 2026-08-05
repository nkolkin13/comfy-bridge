"""MiniMax H3 text-to-video, staged so peak memory holds one big model at a time.

Hand-written from user/default/workflows/video_minimax_h3_t2v.json to show what
generated code plus manual memory management looks like. The workflow's real
graph lives in a subgraph ("Image to Video (MiniMax H3)"); this is that subgraph
flattened, which is what codegen will have to do too.

Why staging is mandatory rather than nice-to-have on this box:

    text encoder  qwen3vl_32b_minimax_h3_nvfp4_awq   (32B, nvfp4)
    diffusion     minimax_h3_fl2va_pruned_int8_convrot
    video vae     minimax_h3_video_vae_fp16
    audio vae     minimax_h3_audio_vae_fp32

against a 24GB 3090 and ~31GB of system RAM. Loading all four at once does not
fit in VRAM, and holding all four *offloaded* does not comfortably fit in RAM
either — ComfyUI's ModelPatcher keeps weights resident in CPU memory when they
are evicted from the GPU. So each stage both unloads from VRAM and drops the
Python reference.

Historical note: an earlier draft used an external chunked VAE decoder; that
turned out to be unnecessary once run_graph() ran under inference_mode (see
Stage C). The utility now lives in examples/vae_chunked.py as a fallback.

This is also why vram_mode is left at "normal": gpu-only would pin the text
encoder, VAEs and intermediates to the GPU (model_management.py:1174-1245) and
OOM instead of degrading.

dynamic_vram=True is *required* here, not an optimisation. It swaps in
ModelPatcherDynamic and sets comfy.memory_management.aimdo_enabled, which is what
lets the int8 diffusion model page weights during sampling. Without it the run
dies partway through step 1 in comfy_kitchen's int8_linear. It has to be passed
explicitly because torch here is 2.6 and upstream gates DynamicVRAM behind
torch >= 2.8 unless asked for directly (main.py:252).

Note the interaction with the staging below: with DynamicVRAM on, VRAM residency
is managed for you, so the `offload()` calls matter less for GPU memory than they
do for *host* RAM — dropping the Python reference is what stops a 32B encoder
sitting in a 31GB system for the whole run.
"""

from __future__ import annotations

import gc

import comfy_bridge

# --------------------------------------------------------------------------
# Startup. Must happen before torch is imported if you want to pin a device.
# --------------------------------------------------------------------------

RT = comfy_bridge.start(device=0, vram_mode="normal", dynamic_vram=True)
assert RT.dynamic_vram, "DynamicVRAM failed to initialise; this graph will OOM"

import torch  # noqa: E402  (only valid after start())

from comfy_bridge import invoke, offload  # noqa: E402


def vram(label: str) -> None:
    import comfy.model_management as mm

    free = mm.get_free_memory(mm.get_torch_device()) / 1024**3
    print(f"  [{label:<22}] {free:5.1f} GB free")


# --------------------------------------------------------------------------
# Graph constants, read off the workflow.
# --------------------------------------------------------------------------

UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

PROMPT = """Realistic live-action cinematic look, award-winning documentary style: practical film photography, a warm scholarly office at golden hour, anamorphic lens, shallow depth of field, subtle film grain, dust motes in window light, restrained naturalistic grading for a premium prestige-documentary feel, minimal and deliberate camera movement.

Scene overview: a well-dressed male professor — tweed blazer, open collar, glasses — sits at a heavy wooden desk in his book-lined office, facing the camera in classic documentary-interview framing. Behind him: floor-to-ceiling bookshelves, a desk lamp, papers, soft window light from the side. He speaks directly and calmly to camera with the gravitas of an expert interviewee, delivering the line: "We really have to be concerned about the capabilities of AI models — for example, I am an AI model, and it's hard to tell."

Storyboard (continuous take with one cut, paced to the dialogue):
[0s–1s] Shot 1: slow push-in from a medium shot, the professor adjusting his glasses, settling, taking a breath before speaking — the quiet beat before an interview answer.
[1s–5s] Shot 2: medium close-up, slightly off-center rule-of-thirds framing, he delivers the line with measured, sincere concern, small natural gestures with one hand, steady eye contact with the lens.
[5s–6s] Shot 3: hold on his face after the final word — a faint, ambiguous pause, one eyebrow slightly raised, letting the line land. No reaction cut, just stillness.

Camera: locked-off tripod feel with a barely perceptible slow push, single clean cut, no dissolves, no handheld shake.

Audio: quiet room tone, faint clock tick, soft HVAC hum, his voice close-miked and warm like a lav mic, no score until the final second — a single low, subtle drone note entering just as he finishes, held into the cut to black.

Lighting: soft key from the window, warm practical lamp fill, gentle falloff into shadowed bookshelves — classic Errol Morris / prestige-doc interview lighting.

No text, subtitles, lower-thirds, logos or watermarks of any kind, no animation or cartoon rendering, no uncanny or overly-CG look — keep photoreal skin texture, natural micro-expressions, and authentic lip sync throughout.
"""

SEED = 596139048893679

# ResolutionSelector: 16:9 @ 0.9 megapixels, multiple of 32.
#WIDTH, HEIGHT = 1280, 736
WIDTH, HEIGHT = 640, 480

# ComfyMathExpression snaps duration to the model's 17k+5 frame grid at 24fps:
#   max(5, round(5 * 24)) = 120 -> 120 + (5 - 120 % 17) % 17 = 124
SECONDS = 3
LENGTH_RAW = round(SECONDS * 24)
LENGTH = LENGTH_RAW + (5 - LENGTH_RAW % 17) % 17
FPS = 24

print('LATENT FRAMES:', LENGTH)

STEPS = 20
DENOISE = 1.0
SCHEDULER = "simple"
SAMPLER = "res_multistep"


@torch.inference_mode()
def run_graph():
    """Mirrors execution.py:751, which wraps ComfyUI's whole prompt execution.

    This one decorator is load-bearing. Without it every node call retains its
    autograd graph, tensors chain together, and VRAM climbs by the full
    activation cost of each node until it OOMs — measured at +7.2 GB per VAE
    tile before this was added, versus 0.11 GB peak after.
    """
    # ---- Stage A: conditioning ------------------------------------------
    # MiniMaxH3ImageToVideo needs clip AND vae together, so this stage is the
    # peak for the 32B encoder. Nothing else is loaded yet.
    vram("start")
    (clip,) = invoke("CLIPLoader", clip_name=CLIP, type="minimax", device="default")
    (video_vae,) = invoke("VAELoader", vae_name=VIDEO_VAE)
    vram("clip + video vae")

    positive, latent = invoke(
        "MiniMaxH3ImageToVideo",
        clip=clip,
        vae=video_vae,
        prompt=PROMPT,
        width=WIDTH,
        height=HEIGHT,
        length=LENGTH,
    )

    # The encoder is done: 32B of weights no longer needed for anything.
    # Dropping the reference matters as much as the VRAM eviction here, because
    # the offloaded copy would otherwise sit in CPU RAM for the whole run.
    offload(clip, video_vae)
    del clip, video_vae
    gc.collect()
    vram("after encoder freed")

    # ---- Stage B: sampling ----------------------------------------------
    (model,) = invoke("UNETLoader", unet_name=UNET, weight_dtype="default")
    (guider,) = invoke("BasicGuider", model=model, conditioning=positive)
    (sigmas,) = invoke(
        "BasicScheduler",
        model=model,
        scheduler=SCHEDULER,
        steps=STEPS,
        denoise=DENOISE,
    )
    (sampler,) = invoke("KSamplerSelect", sampler_name=SAMPLER)
    (noise,) = invoke("RandomNoise", noise_seed=SEED)
    vram("diffusion model")

    sampled, _denoised = invoke(
        "SamplerCustomAdvanced",
        noise=noise,
        guider=guider,
        sampler=sampler,
        sigmas=sigmas,
        latent_image=latent,
    )

    # Sampling is finished; the diffusion model is dead weight during decode.
    offload(model)
    del model, guider, sigmas, sampler, noise
    gc.collect()
    vram("after unet freed")

    # ---- Stage C: video decode ------------------------------------------
    # Reloaded rather than kept from Stage A. That is the D7 "no caching"
    # tradeoff made concrete: a second read off disk, in exchange for not
    # holding the VAE through the whole sampling pass.
    (video_vae,) = invoke("VAELoader", vae_name=VIDEO_VAE)
    # The stock node is enough. An earlier version of this example used an
    # external chunked decoder because decoding appeared to OOM above ~256px —
    # but those measurements predated run_graph()'s inference_mode. MiniMax's VAE
    # tiles internally within one decode() call, so with autograd live it
    # retained every tile's activations. Under inference_mode it decodes 243
    # frames at 864x480 without help. See examples/vae_chunked.py if you ever do
    # exceed it.
    (images,) = invoke("VAEDecode", samples=sampled, vae=video_vae)
    offload(video_vae)
    del video_vae
    gc.collect()
    vram("after video decode")

    # ---- Stage D: audio decode ------------------------------------------
    (audio_vae,) = invoke("VAELoader", vae_name=AUDIO_VAE)
    (audio,) = invoke("VAEDecodeAudio", samples=sampled, vae=audio_vae)
    offload(audio_vae)
    del audio_vae
    gc.collect()
    vram("after audio decode")

    # ---- Stage E: mux and save ------------------------------------------
    (video,) = invoke("CreateVideo", images=images, fps=FPS, audio=audio)
    (saved,) = invoke(
        "SaveVideo",
        video=video,
        filename_prefix="video/MiniMax_H3",
        format="auto",
        # `codec` is a DynamicCombo, not a plain combo: the node reads
        # codec["codec"] and codec.get("encoding"), so it wants a dict keyed by
        # input id even though the UI widget shows the bare string "auto".
        # {"codec": "h264", "encoding": {"encoding": "re-encode", "crf": 23.0}}
        # is the fuller form.
        codec={"codec": "auto"},
    )
    return (saved,)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    RT.progress_callback = lambda event, data, sid: None  # or print(event, data)
    outputs = run_graph()
    print("done:", outputs)
