from torch.utils.data import Dataset

class SlidingWindowDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length, id_col, label_col, stride=384):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id_col = id_col
        self.label_col = label_col
        self.stride = stride 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        code = str(row["code"])
        sample_id = row[self.id_col]
        label = int(row[self.label_col]) if self.label_col in row else -1

        # Tokenizza
        tokens = self.tokenizer.tokenize(code)
        
        max_tokens_limit = self.max_length - 2
        chunks = []
        
        if len(tokens) <= max_tokens_limit:
            chunks = [code]
        else:
            # Sliding Window
            for i in range(0, len(tokens), self.stride):
                chunk_tokens = tokens[i : i + max_tokens_limit]
                chunk_str = self.tokenizer.convert_tokens_to_string(chunk_tokens)
                chunks.append(chunk_str)
                if i + max_tokens_limit >= len(tokens):
                    break
        
        return {
            "id": sample_id,
            "chunks": chunks,
            "label": label,
            "num_chunks": len(chunks)
        }