import gradio as gr
import requests
import pandas as pd

API_BASE = "http://localhost:8000"

def list_models():
    r = requests.get(f"{API_BASE}/models", timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return [], "Registry is not a list. Fix registry.json format."
    names = [f'{m["run_name"]} ({m.get("type","?")})' for m in data]
    return names, data

def parse_run_name(label: str) -> str:
    # "dp_ctgan_... (dp_ctgan)" -> "dp_ctgan_..."
    return label.split(" (")[0].strip()

def train_best():
    r = requests.post(f"{API_BASE}/orchestrate/train_best", timeout=60*60)
    r.raise_for_status()
    return r.json()

def generate(selected, n_rows, device):
    run = parse_run_name(selected)
    r = requests.post(
        f"{API_BASE}/generate",
        params={"run_name": run, "n": int(n_rows), "device": device},
        timeout=60*10
    )
    r.raise_for_status()
    out = r.json()
    df = pd.DataFrame(out["data_preview"])
    return out, df

def visualize(selected, n_synth, device):
    run = parse_run_name(selected)
    r = requests.post(
        f"{API_BASE}/visualize",
        params={"run_name": run, "n_synth": int(n_synth), "device": device},
        timeout=60*10
    )
    r.raise_for_status()
    return r.json()

with gr.Blocks(title="DP Synthetic Tabular Generator (Gradio)") as demo:
    gr.Markdown("## DP Synthetic Tabular Generator — Gradio UI (calls FastAPI)")

    with gr.Row():
        btn_refresh = gr.Button("🔄 Refresh Models")
        btn_train_best = gr.Button("🤖 Train & Pick Best")

    models_dd = gr.Dropdown(label="Select model", choices=[], interactive=True)
    device_dd = gr.Dropdown(label="Device", choices=["cuda", "cpu"], value="cuda")

    out_models_json = gr.JSON(label="Models Registry / List")
    out_train_json = gr.JSON(label="Train Best Output")

    with gr.Row():
        n_rows = gr.Number(label="Generate rows", value=1000, precision=0)
        btn_generate = gr.Button("Generate")
    out_generate_json = gr.JSON(label="Generate Output")
    out_preview_df = gr.Dataframe(label="Preview (first rows)")

    with gr.Row():
        n_synth = gr.Number(label="Viz rows", value=10000, precision=0)
        btn_viz = gr.Button("Visualize")
    out_viz_json = gr.JSON(label="Visualize Output")

    def do_refresh():
        names, raw = list_models()
        return gr.update(choices=names, value=(names[0] if names else None)), raw

    btn_refresh.click(do_refresh, outputs=[models_dd, out_models_json])

    btn_train_best.click(train_best, outputs=out_train_json).then(
        do_refresh, outputs=[models_dd, out_models_json]
    )

    btn_generate.click(generate, inputs=[models_dd, n_rows, device_dd], outputs=[out_generate_json, out_preview_df])

    btn_viz.click(visualize, inputs=[models_dd, n_synth, device_dd], outputs=out_viz_json)

    demo.load(do_refresh, outputs=[models_dd, out_models_json])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
