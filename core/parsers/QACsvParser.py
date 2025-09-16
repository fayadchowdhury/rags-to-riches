from core.parsers.BaseParser import BaseParser
import pandas as pd
import nltk

nltk.download('punkt_tab')

class QACsvParser(BaseParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs

    def read(self, obj):
        self.intermediate = pd.read_csv(obj, skiprows=1, names=["info", "question", "answer"])
        self.intermediate["info"] = self.intermediate["info"].fillna("No extra information given")
        self.filename = obj
        return self.intermediate
    
    def parse(self, obj):
        if not hasattr(self, "intermediate") or self.filename != obj:
            self.read(obj)
        
        self.data = []
        for index, row in self.intermediate.iterrows():
            text = f"Info: {row['info']}\nQuestion: {row['question']}\nAnswer: {row['answer']}"
            self.data.append({
                "file_type": "csv",
                "file_name": "test",
                "marker": index+1,
                "sub_marker": index+1, # Not necessary since we are not chunking
                "text": text,
                "first_10_words": " ".join(text.split()[:10])
            })

        return self.data