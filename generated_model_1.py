
import torch
import torch.nn as nn
from models.mamba import mambalayer

class sum_model(nn.Module):
    def __init__(self):
        super(sum_model, self).__init__()
        print('model building')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.mamba_rna = mambalayer(embedding_dim=641).to(self.device)
        
        self.mamba_ligand = mambalayer(embedding_dim=30).to(self.device)
        
        self.fc = nn.Linear(671, 1).to(self.device)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, rna_feature, ligand_feature):
        rna_feature = rna_feature.float().to(self.device)
        ligand_feature = ligand_feature.float().to(self.device)
        
        rna_out = self.mamba_rna(rna_feature) 
        ligand_out = self.mamba_ligand(ligand_feature)
        
        rna_pooled = torch.mean(rna_out, dim=0)
        ligand_pooled = torch.mean(ligand_out, dim=0)
        
        combined = torch.cat([rna_pooled, ligand_pooled], dim=-1)
        
        output = self.fc(combined)
        output = self.sigmoid(output)
        
        return output
