# predicted as a batch
from params import par
from model import DeepVO
import numpy as np
from PIL import Image
import glob
import os
import time
import torch
from data_helper import get_data_info, ImageSequenceDataset
from torch.utils.data import DataLoader
from helper import eulerAnglesToRotationMatrix

if __name__ == '__main__':

    #videos_to_test = ['04', '05', '07', '10', '09']
    videos_to_test = ['11']
    # Path
    load_model_path = par.load_model_path  # choose the model you want to load
    save_dir = 'result/'  # directory to save prediction answer
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load model
    M_deepvo = DeepVO(par.img_h, par.img_w, par.batch_norm)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        M_deepvo = M_deepvo.cuda()
        M_deepvo.load_state_dict(torch.load(load_model_path))
    else:
        M_deepvo.load_state_dict(torch.load(load_model_path, map_location={'cuda:0': 'cpu'}))
    print('Load model from: ', load_model_path)

    # Data
    n_workers = 1
    seq_len = int((par.seq_len[0] + par.seq_len[1]) / 2)
    overlap = seq_len - 1
    print('seq_len = {},  overlap = {}'.format(seq_len, overlap))
    batch_size = par.batch_size

    fd = open('test_dump.txt', 'w')
    fd.write('\n' + '=' * 50 + '\n')

    for test_video in videos_to_test:
        df = get_data_info(folder_list=[test_video], seq_len_range=[seq_len, seq_len], overlap=overlap, sample_times=1,
                           shuffle=False, sort=False)
        print(f"Original df size: {len(df)}")
        df = df.loc[df.seq_len == seq_len]  # drop last
        print(f"Filtered df size: {len(df)}")
        dataset = ImageSequenceDataset(df, par.resize_mode, (par.img_w, par.img_h), par.img_means, par.img_stds,
                                       par.minus_point_5)
        df.to_csv('test_df.csv')

        valid_samples = []
        for i in range(len(dataset)):
            try:
                _ = dataset[i]  # Try accessing the data
                valid_samples.append(i)  # If no error, keep this index
            except ValueError as e:
                print(f"Skipping index {i}: {e}")

        # Create a filtered dataset with valid indices
        filtered_dataset = torch.utils.data.Subset(dataset, valid_samples)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        gt_pose = np.load('{}{}.npy'.format(par.pose_dir, test_video))  # (n_images, 6)
        # Predict
        M_deepvo.eval()
        has_predict = False
        answer = [[0.0] * 6, ]
        st_t = time.time()
        n_batch = len(dataloader)
        print("n_batch = ", n_batch)

        try :
            for i, batch in enumerate(dataloader):
                print('{} / {}'.format(i, n_batch), end='\r', flush=True)
                print("dataloader length: ", len(dataloader))
                print("batch length: ", len(batch))
                print(f"Batch {i}: Type: {type(batch)}, Length: {len(batch)}")

                # If batch is a tuple or list, print shapes of its elements
                if isinstance(batch, (tuple, list)):
                    for j, item in enumerate(batch):
                        if hasattr(item, "shape"):  # For tensors or NumPy arrays
                            print(f"  Item {j}: shape {item.shape}")
                        else:
                            print(
                                f"  Item {j}: type {type(item)}, length {len(item) if hasattr(item, '__len__') else 'N/A'}")

                _, x, y = batch
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                batch_predict_pose = M_deepvo.forward(x)

                print(f"Batch {i}: batch_predict_pose shape: {batch_predict_pose.shape}")

                # Check for NaNs in the input
                if torch.isnan(x).any():
                    print(f"🚨 NaN detected in x for batch {i}!")

                # Record answer
                fd.write('Batch: {}\n'.format(i))
                for seq, predict_pose_seq in enumerate(batch_predict_pose):
                    for pose_idx, pose in enumerate(predict_pose_seq):
                        fd.write(' {} {} {}\n'.format(seq, pose_idx, pose))

                batch_predict_pose = batch_predict_pose.data.cpu().numpy()
                if i == 0:
                    for pose in batch_predict_pose[0]:
                        # use all predicted pose in the first prediction
                        for i in range(len(pose)):
                            # Convert predicted relative pose to absolute pose by adding last pose
                            pose[i] += answer[-1][i]
                        answer.append(pose.tolist())
                    batch_predict_pose = batch_predict_pose[1:]

                # transform from relative to absolute

                for predict_pose_seq in batch_predict_pose:
                    # predict_pose_seq[1:] = predict_pose_seq[1:] + predict_pose_seq[0:-1]
                    ang = eulerAnglesToRotationMatrix(
                        [0, answer[-1][0], 0])  # eulerAnglesToRotationMatrix([answer[-1][1], answer[-1][0], answer[-1][2]])
                    location = ang.dot(predict_pose_seq[-1][3:])
                    predict_pose_seq[-1][3:] = location[:]

                    # use only last predicted pose in the following prediction
                    last_pose = predict_pose_seq[-1]
                    for i in range(len(last_pose)):
                        last_pose[i] += answer[-1][i]
                    # normalize angle to -Pi...Pi over y axis
                    last_pose[0] = (last_pose[0] + np.pi) % (2 * np.pi) - np.pi
                    answer.append(last_pose.tolist())
        except Exception as e:
            print(f"Error in batch")

        print('len(answer): ', len(answer))
        print('expect len: ', len(glob.glob('{}{}/*.png'.format(par.image_dir, test_video))))
        print('Predict use {} sec'.format(time.time() - st_t))

        # Save answer
        with open('{}/out_{}.txt'.format(save_dir, test_video), 'w') as f:
            for pose in answer:
                if type(pose) == list:
                    f.write(', '.join([str(p) for p in pose]))
                else:
                    f.write(str(pose))
                f.write('\n')

        # Calculate loss
        #
        min_len = min(len(gt_pose), len(answer))
        print ("gt pose length: ", len(gt_pose))
        print ("answer length: ", len(answer))
        print("gt_pose shape: ", gt_pose.shape)
        #print("answer shape: ", answer.shape)
        gt_pose = gt_pose[:min_len]
        answer = answer[:min_len]
        print("gt pose length: ", len(gt_pose))
        print("answer length: ", len(answer))
        print("gt_pose shape: ", gt_pose.shape)
        #
        gt_pose = np.load('{}{}.npy'.format(par.pose_dir, test_video))  # (n_images, 6)
        loss = 0
        for t in range(min_len):
            angle_loss = np.sum((answer[t][:3] - gt_pose[t, :3]) ** 2)
            translation_loss = np.sum((answer[t][3:] - gt_pose[t, 3:6]) ** 2)
            loss += (100 * angle_loss + translation_loss)
        loss /= min_len
        print('Loss = ', loss)
        print('=' * 50)
