from core.parsers.BaseParser import BaseParser
from boilerpy3 import extractors
import nltk

nltk.download('punkt_tab')

class HtmlParser(BaseParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs
        self.extractor = extractors.ArticleExtractor()

    def read(self, obj):
        self.intermediate = self.extractor.get_content_from_file(obj)
        self.filename = obj
        return self.intermediate
    
    def parse(self, obj):
        if not hasattr(self, "intermediate") or self.filename != obj:
            self.read(obj)
        
        self.data = []
        try:
            clean_content = nltk.sent_tokenize(self.intermediate)
            seq_num = 0
            for content in clean_content:
                self.data.append({
                    "file_type": "html",
                    "file_name": self.filename,
                    "text": content,
                    "marker": seq_num
                })
                seq_num += 1
            
        except Exception as e:
            print(f"Error with BoilerPy3 extraction: {e}")
        finally:
            return self.data