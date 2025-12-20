import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, Any

class InferenceDataset(Dataset):
    """
     optimized PyTorch Dataset designed specifically for the Inference/Submission phase.
    
    Key Features:
    - ID Persistence: Preserves sample IDs required for the submission CSV.
    - Deterministic Behavior: No augmentation is applied to ensure reproducible results.
    - Type Safety: Handles mixed-type columns in Pandas dataframes robustly.
    """
    
    def __init__(
        self, 
        dataframe: pd.DataFrame, 
        tokenizer: PreTrainedTokenizer, 
        max_length: int = 384, 
        id_col: str = "id"
    ):
        """
        Args:
            dataframe (pd.DataFrame): Input data containing source code.
            tokenizer (PreTrainedTokenizer): HuggingFace tokenizer instance.
            max_length (int): Maximum sequence length for truncation/padding.
            id_col (str): Name of the column containing unique sample IDs.
        """
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id_col = id_col

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves a sample, tokenizes the code, and returns input tensors with the ID.
        """
        code = str(self.data.loc[idx, "code"])
        id_val = str(self.data.loc[idx, self.id_col])

        encoding = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["id"] = id_val
        
        return item