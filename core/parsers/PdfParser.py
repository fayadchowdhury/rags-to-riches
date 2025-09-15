from core.parsers.BaseParser import BaseParser
from pypdf import PdfReader
import re

class PdfParser(BaseParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs
        self.parser = PdfReader

    def read(self, obj):
        self.intermediate = self.parser(obj)
        self.filename = obj
        return self.intermediate
    
    def parse(self, obj):
        if not hasattr(self, "intermediate") or self.filename != obj:
            self.read(obj)
        num_pages = len(self.intermediate.pages)

        # Ignore <latexit> tags
        latexit_pattern = r"<latexit[^>]*>.*?</latexit>"
        self.data = []
        for i in range(num_pages):
            text = self.intermediate.pages[i].extract_text()
            cleaned_text = re.sub(latexit_pattern, "", text)
            self.data.append({
                "file_type": "pdf",
                "file_name": self.filename,
                "marker": i+1,
                "text": cleaned_text
            })

        return self.data