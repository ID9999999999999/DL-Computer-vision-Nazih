"""Execute every notebook code cell in order and save its actual outputs.

Uses an in-process IPython shell, so no Jupyter socket server is required.
"""
from __future__ import annotations

import os
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


class NotebookShell(InteractiveShell):
    def enable_gui(self, gui=None):
        # The inline backend renders images without a GUI event loop.
        if gui not in (None, 'inline'):
            raise ValueError('This runner supports only inline figures')


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)
    path = root / 'lab05_svm_training.ipynb'
    notebook = nbformat.read(path, as_version=4)
    shell = NotebookShell.instance()
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=True)
        error = result.error_before_exec or result.error_in_exec
        if error is not None:
            print(captured.stdout)
            print(captured.stderr)
            raise RuntimeError(f'Notebook cell {cell.id} failed') from error
        cell.execution_count = result.execution_count
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for rich_output in captured.outputs:
            outputs.append(nbformat.v4.new_output(
                'display_data', data=rich_output.data, metadata=rich_output.metadata,
            ))
        cell.outputs = outputs
        count += 1
        print(captured.stdout, end='')
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f'\nExecuted and saved {count} code cells successfully.')


if __name__ == '__main__':
    main()
