# #!/bin/bash

# set -e

# cd src/iris-flower-classification

# jupyter nbconvert --to notebook --execute iris-feature-pipeline.ipynb
# jupyter nbconvert --to notebook --execute iris-batch-inference-pipeline.ipynb

import os
from nbconvert import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor

def execute_notebook(notebook_path, output_path=None, timeout=600):
    """
    Executes a Jupyter Notebook and optionally saves the executed notebook.
    
    Args:
        notebook_path (str): Path to the input notebook.
        output_path (str, optional): Path to save the executed notebook. Defaults to None.
        timeout (int): Maximum time in seconds for each notebook cell execution.
    """
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = f.read()

        # Configure notebook execution
        ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
        notebook_exporter = NotebookExporter()
        notebook_node = notebook_exporter.from_notebook_node(notebook_content)

        # Execute the notebook
        executed_notebook, _ = ep.preprocess(notebook_node, {'metadata': {'path': os.path.dirname(notebook_path)}})

        # Save the executed notebook if an output path is specified
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(executed_notebook)
                print(f"Executed notebook saved to: {output_path}")

        print(f"Executed notebook: {notebook_path}")
    
    except Exception as e:
        print(f"Error executing notebook {notebook_path}: {e}")
        raise

# Define paths for notebooks
notebooks = [
    "src/iris-flower-classification/iris-feature-pipeline.ipynb",
    "src/iris-flower-classification/iris-batch-inference-pipeline.ipynb"
]

# Execute each notebook
for notebook in notebooks:
    execute_notebook(notebook, output_path=None)
