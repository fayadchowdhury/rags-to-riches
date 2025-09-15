import tiktoken
from core.chunkers.BaseChunker import BaseChunker

class FixedTokenSizeChunker(BaseChunker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = kwargs
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, data):
        max_tokens = self.config.get("max_tokens", 512)
        overlap = self.config.get("overlap", 50)

        chunked_data = []
    
        for entry in data:
            text = entry['text'].split("\n")  # Split text into sentences/lines
            chunks = []
            chunk = []
            current_tokens = 0
            sub_marker = 0
            
            for idx, sentence in enumerate(text):
                sentence_tokens = len(sentence.split())  # Approximation of token count
                if current_tokens + sentence_tokens > max_tokens:
                    # Finalize current chunk
                    chunk_text = " ".join(chunk)
                    chunks.append({
                        **entry,  # Copy the original dictionary fields
                        'text': chunk_text,
                        'sub_marker': sub_marker,
                        'first_10_tokens': " ".join(chunk_text.split()[:10])
                    })
                    sub_marker += 1
                    
                    # Start a new chunk with overlap
                    overlap_sentences = chunk[-overlap:] if overlap < len(chunk) else chunk
                    chunk = overlap_sentences[:]
                    current_tokens = sum(len(s.split()) for s in overlap_sentences)
                
                # Add the current sentence to the chunk
                chunk.append(sentence)
                current_tokens += sentence_tokens
            
            # Add the last chunk
            if chunk:
                chunk_text = " ".join(chunk)
                chunks.append({
                    **entry,
                    'text': chunk_text,
                    'sub_marker': sub_marker,
                    'first_10_tokens': " ".join(chunk_text.split()[:10])
                })
            
            # Append all chunks for this entry to the result
            chunked_data.extend(chunks)

        self.chunks = chunked_data
        return self.chunks