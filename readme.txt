Waste Analysis Package
======================

Canonical documentation lives in README.md.

Setup Quick Guide
-----------------
1) Create environment
     - Windows PowerShell:
         python -m venv venv
         .\venv\Scripts\Activate.ps1
2) Install dependencies
     - pip install -r requirements.txt
3) Optional AI validation
     - create .env with: OPENROUTER_API_KEY=your_key

Startup Quick Guide
-------------------
- Preferred (from parent folder):
    python -m waste_analysis.main
- Direct (from this folder):
    python main.py

Notes
-----
- Data is expected in parent-level Data folder.
- Energy output is sensitivity-based using low/mid/high ranges from modules/energy_config.py:
    biogas_yield = 80, 100, 120
    efficiency = 0.25, 0.35, 0.45
