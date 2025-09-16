from core.parsers.BaseParser import BaseParser
import json
import re

class NotebookParser(BaseParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs

    def read(self, obj):
        with open(obj, 'r', encoding='utf-8') as f:
            self.intermediate = json.load(f)

        self.filename = obj
        return self.intermediate
    
    def parse(self, obj):
        if not hasattr(self, "intermediate") or self.filename != obj:
            self.read(obj)
        self.data = []
        seq_num = 0
        
        # Extract cells
        for cell in self.intermediate.get('cells', []):
            cell_type = cell.get('cell_type')
            if cell_type == 'markdown': # Markdown cell
                md_content = "Text block:\n"
                md_content += "".join(cell.get('source', [])) + "\n"
                # print(md_content)
                self.data.append({
                    "file_type": "ipynb",
                    "file_name": self.filename,
                    "marker": seq_num,
                    "text": md_content
                })
            elif cell_type == 'code':
                code_content = "Code block:\n"
                code_content += "".join(cell.get('source', [])) + "\n"
                # print(code_content)
                # code_source = ''.join(code_content)
                # outputs = []
                code_content += "Output:\n"
                
                # Extract outputs
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'stream':
                        code_content += "".join(output.get('text', [])) + "\n"
                        # print(output_content)
                        # outputs.append(output_content)
                    elif output.get('output_type') == 'execute_result':
                        code_content += "".join(output.get('data', {}).get('text/plain', [])) + "\n"
                        # print(output_content)
                        # outputs.append(output_content)
                    elif output.get('output_type') == 'error':
                        code_content += "Error: ".join(output.get('traceback', [])) + "\n"
                        # print(output_content)
                        # outputs.append('Error: ' + output_content)

                self.data.append({
                    "file_type": "ipynb",
                    "file_name": self.filename,
                    "marker": seq_num,
                    "text": code_content
                })

            seq_num += 1
        
        return self.data
    

def read_code_md_outputs_from_notebook_sequence(file_path):
    # Load the notebook file
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    content = []
    seq_num = 0
    
    # Extract cells
    for cell in notebook.get('cells', []):
        cell_type = cell.get('cell_type')
        if cell_type == 'markdown': # Markdown cell
            md_content = "Text block:\n"
            md_content += "".join(cell.get('source', [])) + "\n"
            # print(md_content)
            content.append({
                "file_type": "ipynb",
                "file_name": file_path,
                "marker": seq_num,
                "text": md_content
            })
        elif cell_type == 'code':
            code_content = "Code block:\n"
            code_content += "".join(cell.get('source', [])) + "\n"
            # print(code_content)
            # code_source = ''.join(code_content)
            # outputs = []
            code_content += "Output:\n"
            
            # Extract outputs
            for output in cell.get('outputs', []):
                if output.get('output_type') == 'stream':
                    code_content += "".join(output.get('text', [])) + "\n"
                    # print(output_content)
                    # outputs.append(output_content)
                elif output.get('output_type') == 'execute_result':
                    code_content += "".join(output.get('data', {}).get('text/plain', [])) + "\n"
                    # print(output_content)
                    # outputs.append(output_content)
                elif output.get('output_type') == 'error':
                    code_content += "Error: ".join(output.get('traceback', [])) + "\n"
                    # print(output_content)
                    # outputs.append('Error: ' + output_content)

            content.append({
                "file_type": "ipynb",
                "file_name": file_path,
                "marker": seq_num,
                "text": code_content
            })

        seq_num += 1
    
    return content