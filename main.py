
from utils import *
from models import *
from unit_test import *
from train import *
from invariant_mpnn import *
from equivariant_mpnn import *

if 'IS_GRADESCOPE_ENV' not in os.environ:
    path = '/scratch/trahman2/qm9'
    target = 0

    # Transforms which are applied during data loading:
    # (1) Fully connect the graphs, (2) Select the target/label
    transform = T.Compose([CompleteGraph(), SetTarget()])
    
    # Load the QM9 dataset with the transforms defined
    dataset = QM9(path, transform=transform)

    # Normalize targets per data sample to mean = 0 and std = 1.
    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean[:, target].item(), std[:, target].item()

RESULTS = {}
DF_RESULTS = pd.DataFrame(columns=["Test MAE", "Val MAE", "Epoch", "Model"])

train_dataset = dataset[:1000]
val_dataset = dataset[1000:2000]
test_dataset = dataset[2000:3000]
print(f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

# Create dataloaders with batch size = 32
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)




# data = train_dataset[10] # one data sample, i.e. molecular graph
# print(f"\nThis molecule has {data.x.shape[0]} atoms, and {data.edge_attr.shape[0]} edges.")

layer = EquivariantMPNNLayer(emb_dim=11, edge_dim=4)
# model = CoordMPNNModel(num_layers=1, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1)
model= FinalMPNNModel(num_layers=1, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1)
dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# # Rotation and translation equivariance unit test for MPNN model
# print(f"Is {type(model).__name__} rotation and translation equivariant? --> {rot_trans_equivariance_unit_test(model, dataloader)}!")

# # Rotation and translation invariance unit test for MPNN layer
print(f"Is {type(layer).__name__} rotation and translation equivariant? --> {rot_trans_equivariance_unit_test(layer, dataloader)}!")



# it = iter(dataloader)
# data = next(it)
# out = model(data)
'''
model_name = type(model).__name__
best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
    model, 
    model_name, # "MPNN w/ Features and Coordinates (Invariant Layers)", 
    train_loader,
    val_loader, 
    test_loader,
    std,
    n_epochs=100
)

RESULTS[model_name] = (best_val_error, test_error, train_time)
df_temp = pd.DataFrame(perf_per_epoch, columns=["Test MAE", "Val MAE", "Epoch", "Model"])
DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)

print(RESULTS)
'''