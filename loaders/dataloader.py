from torch.utils.data import Dataset



class PoseEstimationDataset(Dataset):

        """ SPEED dataset that can be used with DataLoader for PyTorch training. """

        def __init__(self, split='train', speed_root='datasets/', transform_input=None,  transform_gt=None, config=None):

            if split not in {'train', 'validation', 'sunlamp', 'lightbox', 'sunlamp_train', 'lightbox_train'}:
                raise ValueError('Invalid split, has to be either \'train\', \'validation\', \'sunlamp\' or \'lightbox\'')

            if split in {'train', 'validation'}:
                self.image_root = os.path.join(speed_root, 'synthetic', 'images')
                self.mask_root  = os.path.join(speed_root, 'synthetic', 'kptsmap')

                # We separate the if statement for train and val as we may need different training splits
                if split in {'train'}: 
                    with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                        label_list = json.load(f)

                if split in {'validation'}:
                    with open(os.path.join(speed_root, "synthetic", split + '.json'), 'r') as f:
                        label_list = json.load(f)  

            elif split in {'sunlamp_train', 'lightbox_train'}:
                self.image_root = os.path.join(speed_root, split, 'images')
                self.mask_root  = os.path.join(speed_root, split, 'kptsmap')

                with open(os.path.join(speed_root, split, 'train.json'), 'r') as f:
                    label_list = json.load(f)

            else:
                self.image_root = os.path.join(speed_root, split, 'images')

                with open(os.path.join(speed_root, split, 'test.json'), 'r') as f:
                    label_list = json.load(f)

            # Parse inputs
            self.sample_ids = [label['filename'] for label in label_list]
            self.train = (split == 'train') or (split == 'sunlamp_train') or (split == 'lightbox_train')
            self.validation = (split == 'validation') #or (split == 'sunlamp_train') or (split == 'lightbox_train')

            if self.train or self.validation:
                self.labels = {label['filename']: {'q': label['q_vbs2tango_true'], 'r': label['r_Vo2To_vbs_true']} for label in label_list}

            # Load assets
            kpts_mat = scipy.io.loadmat(speed_root + "kpts.mat") 
            self.kpts = np.array(kpts_mat["corners"])   # Spacecraft key-points
            self.cam  = Camera(speed_root)              # Camera parameters
            self.K = self.cam.K

            self.K[0, :] *= ((config["cols"])/1920)
            self.K[1, :] *= ((config["rows"])/1200)    

            # Transforms for the tensors inputed to the network
            self.transform_input  = transform_input
            self.col_factor_input = ((config["cols"])/1920)
            self.row_factor_input = ((config["rows"])/1200)   

        def __len__(self):
            return len(self.sample_ids)

        def __getitem__(self, idx):
            
            # Load image
            sample_id = self.sample_ids[idx]

            img_name  = os.path.join(self.image_root, sample_id)
            pil_image = cv2.imread(img_name)
            torch_image  = self.transform_input(pil_image)

            y = sample_id

            # For validation, just load the gt pose
            if self.validation:
                q0, r0 = self.labels[sample_id]['q'], self.labels[sample_id]['r']

            # For training, we need more stuff
            if self.train:
                q0, r0 = self.labels[sample_id]['q'], self.labels[sample_id]['r']

                ## Before computing the true key-point positions we need some transformations
                q  = quat2dcm(q0)
                r = np.expand_dims(np.array(r0),axis=1)
                r = q@(r)

                ## Spacecraft kpts placed in front of the camera
                kpts_cam = q.T@(self.kpts+r)

                ## Project key-points to the camera
                kpts_im = self.K@(kpts_cam/kpts_cam[2,:])
                kpts_im = np.transpose(kpts_im[:2,:])

                heatmap_id  = sample_id.split(".jpg")[0] + ".npz"
                data_loaded = np.load(os.path.join(self.mask_root, heatmap_id),allow_pickle=True)
                
                pil_heatmap  = data_loaded.f.arr_0.astype(np.float32)
                visible_kpts = data_loaded.f.arr_3.astype(np.bool8)
            
                torch_heatmap  = self.transform_input(pil_heatmap)

        
            sample = dict()
            if self.train:
                sample["image"]      = torch_image
                sample["heatmap"]    = torch_heatmap
                sample["kpts_3Dcam"] = kpts_cam.astype(np.float32)
                sample["kpts_2Dim"]  = kpts_im.astype(np.float32)
                sample["visible_kpts"] = visible_kpts

                sample["q0"]         = np.array(q0).astype(np.float32)  
                sample["r0"]         = np.array(r0).astype(np.float32)   
                sample["y"]          = y

            if self.validation:
                sample["image"]      = torch_image
                sample["q0"]         = np.array(q0).astype(np.float32)  
                sample["r0"]         = np.array(r0).astype(np.float32)   
                sample["y"]          = y                

            else:
                sample["image"]    = torch_image
                sample["y"] = y


            return sample