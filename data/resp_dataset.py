import pandas as pd
import torch

class RespDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 0]
        audio_fts = torch.tensor(self.data.iloc[idx, [x for x in range(1,26)]].tolist(), dtype=torch.float32)   

        target_str = self.data.iloc[idx, 26]

        diagnosis_dict = {'healthy': 0,
                          'positive_mild': 1,
                          'no_resp_illness_exposed': 0,
                          'resp_illness_not_identified': 0,
                          'positive_moderate': 1,
                          'recovered_full': 0,
                          'positive_asymp': 1}

        target = diagnosis_dict[target_str]

        return audio_fts, target