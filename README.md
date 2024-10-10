# SSY340Project
Face + emotion detection for the cource SSY340 at chalmers university

activate env in terminal: ". .\.venv\Scripts\activate.ps1" or "./env/Scripts/activate.ps1"
Save env to requirements: "pip freeze > requirements.txt"
install packages from txt: "pip install -r requirements.txt" (add pytorch channel --extra-index-url https://download.pytorch.org/whl/cu118)
Run bottleneck analysis: "python -m torch.utils.bottleneck .\main.py"