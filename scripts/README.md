# Scripts

Utility scripts for development and debugging.

## Available Scripts

### run_simulation.py

🏥 **Launches the Real-Time CTG Simulation Dashboard**

```bash
# Using the launcher script
python scripts/run_simulation.py

# Or directly with streamlit
streamlit run src/ui/simulation_app.py
```

This dashboard provides:
- Real-time simulation of 8 patients
- Live CTG plotting with FHR and UC signals
- Event injection for training scenarios (Late decels, Sinusoidal, etc.)
- AI-powered analysis with MOMENT model
- Hebrew/English bilingual interface

**Requirements:** Make sure `streamlit` is installed: `pip install streamlit plotly`

### visualize_preprocessing.py

Visualizes the preprocessing pipeline stages for CTG signals.

```bash
python scripts/visualize_preprocessing.py
```

This script:
- Loads a sample CTG record
- Shows raw vs preprocessed FHR signal
- Displays gap filling and spike removal effects
- Generates comparison plots
