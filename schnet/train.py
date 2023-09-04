import torch
import torchmetrics
import schnetpack as spk
import schnetpack.transform as trn
import pytorch_lightning as pl
import os

from schnetpack.data import AtomsDataModule, AtomsDataFormat
from pytorch_lightning.strategies import DDPStrategy

model_dir = "./ec-model"
batch_size = 1
cutoff = 7.5
database_file_name = './ec-test.db'
n_atom_basis = 128
epochs = 10
num_workers = 1
gpus = [0]
num_nodes = 1

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

transforms = [
    trn.ASENeighborList(cutoff=cutoff),
    trn.CastTo32()
]

dataset = AtomsDataModule(
    datapath=database_file_name,
    load_properties=['charges'],
    transforms=transforms,
    distance_unit='Ang',
    property_units={'charges': '_e'},
    batch_size=batch_size,
    num_train=2,
    num_val=1,
    num_test=1,
    num_workers=num_workers,
    format=AtomsDataFormat.ASE
)

pairwise_distance = spk.atomistic.PairwiseDistances()
radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis,
    n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)

pred_charges = spk.atomistic.Atomwise(
    n_in=n_atom_basis,
    aggregation_mode=None,
    per_atom_output_key='charges'
)

nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_charges],
    postprocessors=[
        trn.CastTo64()
    ]
)

output_charges = spk.task.ModelOutput(
    name='charges',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.0,
    metrics={
        'MAE': torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_charges],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={'lr': 1e-4}
)

logger = pl.loggers.TensorBoardLogger(save_dir=model_dir)
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join(model_dir, 'best_inference_model'),
        save_top_k=1,
        monitor='val_loss'
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir=model_dir,
    max_epochs=epochs,
    num_nodes=num_nodes,
    accelerator='gpu',
    devices=gpus,
    detect_anomaly=True
)

trainer.fit(task, datamodule=dataset)
