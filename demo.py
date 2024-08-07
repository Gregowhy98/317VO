import argparse
from src.utils import *
import torch


# ----------------------------- #
#     Initialization            #
# ----------------------------- #

parser = argparse.ArgumentParser(description="Gre VO Net")
parser.add_argument("-c", "--cfg", metavar="DIR", help="Path to the configuration file", required=True)
args = parser.parse_args()

# Parse the config file
config = utils.load_config(args.cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # device = config["device"]

# Instantiate the network
hourglass  = large_hourglass.get_large_hourglass_net(heads={'hm_c':11, 'depth': 11}, config["num_stacks"]).to(device)

# Load pretrained weights
if config["path_pretrain"]:
    model_dict = torch.load(config["path_pretrain"])
    hourglass.load_state_dict(model_dict, strict=False)   # strict=true

print("\n------------------  Demo started  -------------------\n")
print("  -- Using config from:\t", args.cfg)
print("  -- Using weights from:\t", config["path_pretrain"])
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

val_dataset = speedplus.PyTorchSatellitePoseEstimationDataset(split="validation",
              speed_root=config["root_dir"], transform_input=tforms, config=config)

val_loader    = torch.utils.data.DataLoader(val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=config["num_workers"],
                drop_last=False,
                pin_memory=False)

# ----------------------------- #
#        Load Data              #
# ----------------------------- #

k_mat_input = utils.get_kmat_scaled(config, device) # Intrinsic matrix
dist_coefs  = utils.get_coefs(config, device) # Distortion coefficients
kpts_world  = utils.get_world_kpts(config,device) # Spacecraft key-points

#--------------------------------------
#               Test Loop
#--------------------------------------

with torch.no_grad():
    hourglass.eval()
    speed_total_m, speed_t_total_m, speed_r_total_m = test_loop(val_loader,
                                hourglass, 
                                kpts_world, 
                                k_mat_input, 
                                device, 
                                config)