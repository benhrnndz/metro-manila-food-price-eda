import os
import sys
import io
import ast
import base64
import contextlib
import nbformat as nbf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# Set backend to non-interactive Agg
matplotlib.use('Agg')
os.environ['MPLCONFIGDIR'] = os.path.abspath('.matplotlib')
os.makedirs('.matplotlib', exist_ok=True)

nb_path = 'EDA_Food_Price_Monitoring.ipynb'
nb = nbf.read(nb_path, as_version=4)

globals_dict = {
    '__name__': '__main__',
    '__doc__': None,
}

execution_count = 1

for cell_idx, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    
    code = cell.source.strip()
    cell.execution_count = execution_count
    cell.outputs = []
    
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    
    # Check if the last node is an expression
    try:
        parsed = ast.parse(code)
    except Exception as e:
        print(f"Cell {cell_idx} syntax error: {e}")
        continue
    
    last_expr = None
    exec_code = code
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        last_expr_node = parsed.body[-1]
        # split off the last expression to evaluate it
        expr_source = ast.unparse(last_expr_node.value)
        body_without_last = parsed.body[:-1]
        if body_without_last:
            exec_code = ast.unparse(body_without_last)
        else:
            exec_code = ""
        last_expr = expr_source

    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        try:
            if exec_code:
                exec(exec_code, globals_dict)
            
            expr_val = None
            if last_expr:
                expr_val = eval(last_expr, globals_dict)
        except Exception as e:
            err_msg = f"Error in cell {cell_idx}: {type(e).__name__}: {e}"
            print(err_msg, file=stderr_buf)
            print(err_msg)
    
    # 1. Capture stdout
    stdout_text = stdout_buf.getvalue()
    if stdout_text:
        cell.outputs.append(nbf.v4.new_output(
            output_type='stream',
            name='stdout',
            text=stdout_text
        ))
        
    # 2. Capture stderr
    stderr_text = stderr_buf.getvalue()
    if stderr_text:
        cell.outputs.append(nbf.v4.new_output(
            output_type='stream',
            name='stderr',
            text=stderr_text
        ))
        
    # 3. Capture Matplotlib Figures
    fig_nums = plt.get_fignums()
    if fig_nums:
        for fnum in fig_nums:
            fig = plt.figure(fnum)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', dpi=120, bbox_inches='tight')
            img_buf.seek(0)
            img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
            
            cell.outputs.append(nbf.v4.new_output(
                output_type='display_data',
                data={
                    'image/png': img_base64,
                    'text/plain': '<Figure size ... with ... Axes>'
                },
                metadata={}
            ))
        plt.close('all')
        
    # 4. Capture last expression result (e.g. DataFrame)
    if expr_val is not None:
        data = {}
        if hasattr(expr_val, '_repr_html_'):
            data['text/html'] = expr_val._repr_html_()
        data['text/plain'] = repr(expr_val)
        
        cell.outputs.append(nbf.v4.new_output(
            output_type='execute_result',
            execution_count=execution_count,
            data=data,
            metadata={}
        ))
        
    execution_count += 1
    print(f"Executed cell {cell_idx} (Execution count: {execution_count - 1})")

with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully executed and saved {nb_path} with all outputs embedded!")
