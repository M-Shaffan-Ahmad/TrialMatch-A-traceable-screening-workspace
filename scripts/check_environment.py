"""Read-only hardware/dependency check; no downloads or model loading."""
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

info = {"python": sys.version.split()[0], "platform": platform.platform(),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "latex_available": shutil.which("pdflatex") is not None,
        "dependencies": {name: importlib.util.find_spec(name) is not None for name in
                         ("numpy", "pyarrow", "huggingface_hub", "torch", "transformers", "peft", "streamlit")}}
if info["dependencies"]["torch"]:
    import torch
    info["cuda_available"] = torch.cuda.is_available()
    info["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
Path("results").mkdir(exist_ok=True)
Path("results/environment.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info, indent=2))
