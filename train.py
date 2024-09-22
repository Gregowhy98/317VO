
import argparse
from src import utils
import torch
import os
from tqdm import tqdm
import pathlib
import numpy as np
import cv2
# from torch.utils import tensorboard

from torchvision import transforms

from data.mydataset import FeatureFusionDataset

from models.mynet import GreVONet
from models.superpoint import SuperPointFrontend, SuperPointNet

# ----------------------------- #
#     Initialization            #
# ----------------------------- #

# parser = argparse.ArgumentParser(description="GreVO Net")
# parser.add_argument("-c", "--cfg", metavar="DIR", help="Path to the configuration file", required=True)
# args = parser.parse_args()
# config = utils.load_config(args.cfg)

config = utils.load_config("/home/wenhuanyao/317VO/configs/train_configs.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Create the direcctories for the tensorboard logs
# path_logs = os.path.join(config["path_results"],args.cfg,"logs")
# pathlib.Path(path_logs).mkdir(parents=True, exist_ok=True)
# writer  = tensorboard.writer.SummaryWriter(path_logs)

# initialize the network
model = GreVONet().to(device)   # todo: add the configs to the model

if config["pretrained_model"]:
    model_dict = torch.load(config["pretrained_model_dir"])
    model.load_state_dict(model_dict,strict=False)

print("\n--------------  Training started  -------------------\n")
# print("  -- Using config from:\t", args.cfg)
print("  -- Using weights from:\t", config["pretrained_model_dir"])
print("  -- Saving weights to:\t", config["path_results"])
print("\n-----------------------------------------------------\n")

# ----------------------------- #
#           Transforms          #
# ----------------------------- #

tforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((config["rows"], config["cols"]))
    ])

# ----------------------------- #
#           Loaders             #
# ----------------------------- #

train_dataset = FeatureFusionDataset(config['dataset_root_dir'], use='train', 
                                     if_sp=False, raw_seg=False, weighted_seg=True)
    # todo: add the transforms

val_dataset = FeatureFusionDataset(config['dataset_root_dir'], use='val',
                                    if_sp=False, raw_seg=False, weighted_seg=True)
    
train_loader = torch.utils.data.DataLoader(train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=True,
                pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                batch_size=config["batch_size_val"],
                shuffle=False,
                num_workers=config["num_workers"],
                drop_last=False,
                pin_memory=False)

# ----------------------------- #
#        Optimizer/Loss         #
# ----------------------------- #

optim_params = [ {"params": model.parameters(),
                  "lr": config["lr"]}]
optimizer = torch.optim.Adam(optim_params)
mse_loss = torch.nn.MSELoss()

# ----------------------------- #
#        Load SP                #
# ----------------------------- #

best_val_score = 1e16

# superpoint
weights_path = config["superpoint"]["model_path"]
sp = SuperPointNet()
sp.load_state_dict(torch.load(weights_path))
sp.cuda()
sp.eval()
# sp_front = SuperPointFrontend(weights_path, 4, 0.015, 0.7, True)

# ----------------------------- #
#        Train/Val Loop         #
# ----------------------------- #

for epoch in range(config["start_epoch"], config["num_epochs"]):
    print("Epoch: ", epoch, "\n")
    model.train()
    train_loss = 0
    
    # ----------------------------- #
    #        Train Epoch            #
    # ----------------------------- #

    for i, data in enumerate(tqdm(train_loader, desc="Training", ncols=100)):
        # Load the data
        img = data['raw_img'].to(device)
        gt = data['weighted_seg'].to(device)

        # pseduo ground truth
        sp_img = sp.sp_process(data['raw_img'])
        semi, desc = sp(sp_img)
        
        # Forward pass
        pred = model(img)
        
        # Initialize variables
        total_loss = 0
        loss_combi = 0
        
        # Compute the loss term
        kpts_loss = mse_loss(pred, semi)
        # loss_combi = r_err + t_err

        # Update loss
        train_loss += loss.item()
        total_loss +=  loss_hm + 1e-2*loss_pnp + 1e-2*loss_3d
        
        # # Update tensorboard logs
        # if not i%config["save_tensorboard"]:
        #     dict_writer["heatmap",level_id] = heatmap_pred
        #     dict_writer["depth",level_id] = depth_pred
        #     dict_writer["loss_hm",level_id] = loss_hm.item()
        #     dict_writer["total_loss",level_id] = total_loss.item()
    
        # Compute loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
#         if not i%config["save_tensorboard"]:
#             utils.update_writer(writer,dict_writer, i + len(train_loader)*epoch, config)

    # Compute the average loss
    train_loss /= len(train_loader)

#     # Log the loss
#     writer.add_scalar('Loss/train', train_loss, epoch)
    
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

#         writer.add_scalar("Validation Pose Score",  val_score, epoch)
#         writer.add_scalar("Validation Translation Score",  val_score_t, epoch)
#         writer.add_scalar("Validation Rotation Score",  val_score_r, epoch)
#         print("Validation Score: \n", val_score)

        if val_score < best_val_score or not epoch%config["save_epoch"]:
            best_val_score = val_score

            string_model = "epoch_" + str(epoch) + "_" + str(best_val_score) + "model_seg.pth"
            torch.save(model.state_dict(),  os.path.join(path_checkpoints, string_model))

            if config["save_optimizer"]:
                string_optimizer = "epoch_" + str(epoch) + "_" + str(best_val_score) + "optimizer.pth"
                torch.save(optimizer.state_dict(),  os.path.join(path_checkpoints, string_optimizer))

        if epoch+1 == config["total_epochs"]:
            torch.save(model.state_dict(),  os.path.join(path_checkpoints, "last_epoch_" + str(epoch) +"model_seg.pth"))
else:
    torch.save(model.state_dict(),  os.path.join(path_checkpoints, "init.pth"))