from streamlit.runtime.uploaded_file_manager import UploadedFile
import tempfile
import os
from contextlib import contextmanager
from typing import Generator

@contextmanager
def create_temp_file(file: UploadedFile) -> Generator[str, None, None]:
    '''
    Take the uploaded file and save to a temporary file for parsers to read
    Return the path to the temporary file
    '''
    extension = file.name.split(".")[-1]
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}")
    try:
        tmp_file.write(file.read())
        tmp_file.close()
        yield tmp_file.name
    finally:
        os.remove(tmp_file.name)