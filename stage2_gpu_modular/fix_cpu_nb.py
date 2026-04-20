import json

# Path to the CPU Notebook
nb_path = r'c:\MLA\CelebA_Local\CelebA_Training_Notebook.ipynb'

# Read the notebook
with open(nb_path, 'r') as f:
    notebook = json.load(f)

# Loop through and delete verbose=True
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if "verbose=True" in line:
                # Remove the line entirely (or replace it if it's inline)
                line = line.replace(",\n    verbose=True", "")
                line = line.replace("verbose=True", "")
            new_source.append(line)
        cell['source'] = new_source

# Save the notebook
with open(nb_path, 'w') as f:
    json.dump(notebook, f, indent=1)
