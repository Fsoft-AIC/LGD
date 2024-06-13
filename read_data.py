import pickle
import torch

# with open("data/grasp-anything++/unseen/grasp_instructions/0ba05c786580d26941b01d3a05ae70c75272a2bdd03df13c7c696fd68f158dc8_0_0.pkl", 'rb') as f:
#     data = pickle.load(f)
#     print(data)

with open("data/grasp-anything++/unseen/grasp_label/0a4e30a3ff80cf312aa71aa43f59bc628fc2dfb28d9e9c5fd1498e8c3bb25271_0_0.pt", 'rb') as f:
    pos_grasps = torch.load(f)
    print(pos_grasps)