
import argparse
from src import utils
import torch
import os
import tqdm
import pathlib
import numpy as np
from torch.utils import tensorboard

from torchvision import transforms

from data.mydataset import FeatureFusionDataset

from models import large_hourglass
from models.superpoint import SuperPointNet
from models.abandon.unet import UNet
from models.mynet import GreVONet

# ----------------------------- #
#     Initialization            #
# ----------------------------- #

parser = argparse.ArgumentParser(description="Gre VO Net")
parser.add_argument("-c", "--cfg", metavar="DIR", help="Path to the configuration file", required=True)
args = parser.parse_args()

config = utils.load_config(args.cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     # device = config["device"]

# Create the direcctories for the tensorboard logs
path_logs = os.path.join(config["path_results"],args.cfg,"logs")
pathlib.Path(path_logs).mkdir(parents=True, exist_ok=True)

# Instantiate the tensorboard writer
writer  = tensorboard.writer.SummaryWriter(path_logs)

# initialize the network
model = GreVONet(config).to(device)

# load the pretrained model
if config["path_pretrain"]:
    model_dict = torch.load(config["path_pretrain"])
    model.load_state_dict(model_dict,strict=False)

print("\n--------------  Training started  -------------------\n")
print("  -- Using config from:\t", args.cfg)
print("  -- Using weights from:\t", config["path_pretrain"])
print("  -- Saving weights to:\t", config["path_results"])
print("\n-----------------------------------------------------\n")

# ----------------------------- #
#           Transforms          #
# ----------------------------- #

# These are applied in the data loader
tforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((config["rows"], config["cols"]))
    ])

# ----------------------------- #
#           Loaders             #
# ----------------------------- #

train_dataset = FeatureFusionDataset(img_folder=config["root_dir"], transform=tforms)

val_dataset = FeatureFusionDataset(img_folder=config["root_dir"], transform=tforms)
    
train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=True,
                pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=config["num_workers"],
                drop_last=False,
                pin_memory=False)

# ----------------------------- #
#        Optimizer/Loss         #
# ----------------------------- #

optim_params = [ {"params": hourglass.parameters(),
                  "lr": config["lr"]}]
optimizer = torch.optim.Adam(optim_params)
mse_loss = torch.nn.MSELoss()

# ----------------------------- #
#        Load Data              #
# ----------------------------- #

k_mat_input = utils.get_kmat_scaled(config, device) # Intrinsic matrix

# Dictionary used to update writer
dict_writer = {}
dict_writer["kpts_world"]  = kpts_world[0]
dict_writer["k_mat_input"] = k_mat_input[0]

best_val_score = 1e16

# ----------------------------- #
#        Train/Val Loop         #
# ----------------------------- #

for epoch in range(config["start_epoch"], config["num_epochs"]):
    print("Epoch: ", epoch, "\n")
    GreVONet.train(True)
    train_loss = 0
    
    # ----------------------------- #
    #        Train Epoch            #
    # ----------------------------- #

    for i, data in enumerate(tqdm(train_loader, desc="Training", ncols=100)):
        # Load the data
        img = data[].to(device)
        gt = data[].to(device)

        # Forward pass
        pred = model(img)
        pred_0 = pred[0]
        pred_1 = pred[1]
        
        # Initialize variables
        total_loss = 0
        loss_combi = 0
        r_err = 0
        t_err = 0
        
        # Compute the loss term
        r_err = mse_loss(pred_0, gt_0)
        t_err = mse_loss(pred_1, gt_1)
        loss_combi = r_err + t_err



        # Update loss
        train_loss += loss.item()
        # total_loss +=  loss_hm + 1e-2*loss_pnp + 1e-2*loss_3d
        
        # Update tensorboard logs
        if not i%config["save_tensorboard"]:
            dict_writer["heatmap",level_id] = heatmap_pred
            dict_writer["depth",level_id] = depth_pred
            dict_writer["loss_hm",level_id] = loss_hm.item()
            dict_writer["total_loss",level_id] = total_loss.item()
    
        # Compute loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if not i%config["save_tensorboard"]:
            utils.update_writer(writer,dict_writer, i + len(train_loader)*epoch, config)

    # Compute the average loss
    train_loss /= len(train_loader)

    # Log the loss
    writer.add_scalar('Loss/train', train_loss, epoch)
    
    # ----------------------------- #
    #         Eval Loop             #
    # ----------------------------- #
    
    with torch.no_grad():
        model.eval()
        val_score, val_score_t, val_score_r = utils.eval_loop(val_loader,
                                                    aug_intensity_val,
                                                    hourglass,
                                                    kpts_world,
                                                    k_mat_input,
                                                    device,
                                                    config)

        writer.add_scalar("Validation Pose Score",  val_score, epoch)
        writer.add_scalar("Validation Translation Score",  val_score_t, epoch)
        writer.add_scalar("Validation Rotation Score",  val_score_r, epoch)
        print("Validation Score: \n", val_score)

        if val_score < best_val_score or not epoch%config["save_epoch"]:
            best_val_score = val_score

            string_model = "epoch_" + str(epoch) + "_" + str(best_val_score) + "model_seg.pth"
            torch.save(hourglass.state_dict(),  os.path.join(path_checkpoints, string_model))

            if config["save_optimizer"]:
                string_optimizer = "epoch_" + str(epoch) + "_" + str(best_val_score) + "optimizer.pth"
                torch.save(optimizer.state_dict(),  os.path.join(path_checkpoints, string_optimizer))

        if epoch+1 == config["total_epochs"]:
            torch.save(hourglass.state_dict(),  os.path.join(path_checkpoints, "last_epoch_" + str(epoch) +"model_seg.pth"))
else:
    torch.save(hourglass.state_dict(),  os.path.join(path_checkpoints, "init.pth"))