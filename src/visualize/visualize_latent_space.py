import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import sklearn.manifold
import torch
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.parser.visualize import parser

from src.utils.get_model_and_data import get_model_and_data

import src.utils.fixseed  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = '3'


plt.switch_backend('agg')


if __name__ == '__main__':
    # parse options

    parameters, folder, checkpointname, epoch = parser()

    model, datasets = get_model_and_data(parameters)
    dataset = datasets["train"]
    model.eval()
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)

    nexemple = 100
    gamma = []
    beta = []
    z = []
    labels = []
    generats = []
    
    print("Evaluating model..")
    keep = {"x": [], "y": [], "di": []}

    num_classes = dataset.num_classes
    # num_classes = 1
    
    for label in tqdm(range(num_classes)):
        a = torch.ones([1,1],dtype=int)
        a[0,0] = int(label)
        xcp, ycp, mask,length = dataset.get_label_sample_all(a.numpy())
        max_num = 50
        if  xcp.size()[0] > max_num:
            maxclip = max_num
        else:
            maxclip = xcp.size()[0]
        gen = {"x": xcp[:maxclip,:,:,:].to(model.device),
               "y": ycp.squeeze(0)[:maxclip].to(model.device),
               "mask": mask.repeat(maxclip,1),
               "lengths": length.to(model.device),
               "output": xcp[:maxclip,:,:,:].to(model.device)}
        h=model(gen)
        gammacp = h["gamma"].data.cpu().numpy()
        betacp = h["beta"].data.cpu().numpy()
        zcp = h["z"].data.cpu().numpy()[:maxclip,:]
        ycp = h["y"].data.cpu().numpy()
        gamma.append(gammacp)
        beta.append(betacp)
        z.append(zcp)
        labels.append(ycp)
        
    gamma = np.array(gamma)
    beta = np.array(beta)
    z = np.array(z)
    nclasses, nexemple, latent_dim = gamma.shape
    labels = np.array(labels)
    all_gamma = np.concatenate(gamma)
    all_beta = np.concatenate(beta)
    all_z = np.concatenate(z)
    nall_latents = len(all_beta)

    # import ipdb; ipdb.set_trace()
    print("Computing tsne..")

    all_input_before = all_z
    all_input_after = all_z*all_gamma + all_beta
    # tsne = TSNE(n_components=2)
    # all_vizu_concat = tsne.fit_transform(all_input)
    # import ipdb; ipdb.set_trace()
    # feats = tuple(np.argsort(all_latents.var(0))[::-1][:2])
    feats = tuple(np.argsort(all_input_after.min(0)-all_input_after.max(0))[::-1][:2] )
    all_vizu_concat = sklearn.manifold.TSNE(n_components=2).fit_transform(all_input_before)
    #all_vizu_concat = all_input_after[:,feats]
    all_vizu_vectors = all_vizu_concat[:nall_latents]

    vizu_vectors = all_vizu_vectors.reshape(nclasses, nexemple, 2)
    
    print("Plotting..")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    #colors = ['gray','b','g','r','c','m','y','k','pink','purple','silver','salmon']
    for label in tqdm(range(num_classes)):
        color = colors[label]
        plt.scatter(vizu_vectors[label,:,0],vizu_vectors[label,:,1], color=color)
        
    plt.savefig("tsne_all_before.png")
    plt.close()

