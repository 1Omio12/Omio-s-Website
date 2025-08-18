# OpenMC Web Simulator + Online Code Editor

This app provides:
- XML upload workflow to run OpenMC on the server and visualize results
- Online Python code editor that executes user code to generate OpenMC XML, then runs the simulation
- Live logs, geometry plots, and tally chart on the results page
- Optional AI suggestions for code via free/local providers

Prerequisites
- Linux, Python 3.10+
- Redis server for Celery: `sudo apt install redis-server && sudo systemctl enable --now redis-server`
- OpenMC HDF5 cross-sections on disk

Setup
1) Create venv and install dependencies
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure cross-sections (recommended via env)
```
export OPENMC_XS_JEFF33=/path/to/jeff-3.3-hdf5/cross_sections.xml
export OPENMC_XS_ENDFBVIII0=/path/to/endfb-viii.0-hdf5/cross_sections.xml
export OPENMC_XS_ENDFBVII1=/path/to/endfb-vii.1-hdf5/cross_sections.xml
```

3) Start services (separate shells)
```
source .venv/bin/activate
celery -A app.celery worker --loglevel=INFO
```
```
source .venv/bin/activate
python app.py
```

4) Open `http://localhost:5000`
- Upload XMLs on the home page
- Use the code editor at `/editor`

Optional: AI suggestions
- Local Ollama (free):
```
export AI_PROVIDER=ollama
export AI_MODEL=llama3.1
ollama serve  # ensure http://localhost:11434
```
- Hugging Face Inference API:
```
export AI_PROVIDER=huggingface
export HF_API_KEY=hf_xxx
export AI_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

Notes
- User code must write `materials.xml`, `geometry.xml`, and `settings.xml` in the working directory.
- Code runs with a timeout and no interactivity; add sandboxing before exposing publicly.