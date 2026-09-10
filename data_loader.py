import torch
from torch.utils.data import Dataset, DataLoader
import math

class CFDAerospaceDataset(Dataset):
    """
    Simulated CFD Dataset containing boundary condition data streams.
    In a real scenario, this would load CSVs/HDF5s exported from OpenFOAM or Ansys Fluent.
    Feature Maps: [Inlet_Temp(K), Ambient_Pressure(atm), Coolant_Flow_Rate(L/min), Heat_Flux_Density(W/cm^2)]
    """
    def __init__(self, num_samples=1000, n_features=4):
        # We generate synthetic aerodynamic data typical for a Hydrogen Fuel Cell
        
        # 1. Inlet Temp: 320 to 360 Kelvin
        inlet_temp = torch.rand(num_samples) * 40 + 320
        # 2. Ambient Pressure (Altitude effect): 0.5 to 1.2 atm
        pressure = torch.rand(num_samples) * 0.7 + 0.5
        # 3. Coolant Flow Rate: 5 to 15 L/min
        flow_rate = torch.rand(num_samples) * 10 + 5
        # 4. Heat Flux Density from Fuel Cell: 1.0 to 3.5 W/cm^2
        heat_flux = torch.rand(num_samples) * 2.5 + 1.0
        
        base_features = [inlet_temp, pressure, flow_rate, heat_flux]
        
        # Add random noise-features for remaining qubit slots to demonstrate scaling
        for i in range(n_features - 4):
            base_features.append(torch.rand(num_samples))

        self.raw_data = torch.stack(base_features, dim=1)
        
        # Q-Circuit Data Preparation: 
        # We scale features seamlessly directly to [0, Pi] for the Qubit Angle Embedding matrices.
        self.max_vals = self.raw_data.max(dim=0)[0]
        self.scaled_data = (self.raw_data / self.max_vals) * math.pi
        
    def __len__(self):
        return len(self.raw_data)
        
    def __getitem__(self, idx):
        # We return both the scaled features (for Quantum gates) and raw data (For physics calculation)
        return self.scaled_data[idx], self.raw_data[idx]

def get_cfd_dataloader(batch_size=8, samples=500, n_features=4):
    dataset = CFDAerospaceDataset(num_samples=samples, n_features=n_features)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
