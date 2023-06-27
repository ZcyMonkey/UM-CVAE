import os

import matplotlib.pyplot as plt
import torch
import numpy as np
import src.utils.rotation_conversions as geometry

from src.utils.get_model_and_data import get_model_and_data
from src.models.get_model import get_model

from src.parser.generate import parser
import src.utils.fixseed  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
plt.switch_backend('agg')


def generate_actions(beta, model, dataset, epoch, params, folder, num_frames=64,
                     durationexp=False, vertstrans=True, onlygen=False, nspa=10, inter=False, writer=None):
    """ Generate & viz samples """

    # visualize with joints3D
    model.outputxyz = True
    # print("remove smpl")
    model.param2xyz["jointstype"] = "vertices"

    print(f"Visualization of the epoch {epoch}")

    fact = params["fact_latent"]
    num_classes = dataset.num_classes
    all_classes = torch.arange(num_classes)
    output = []
    for num in range(num_classes):
        classes = all_classes[num:num+1]
        gendurations = torch.tensor([num_frames], dtype=int)
        gendurations = gendurations.repeat((nspa, 1))


        print("Computing the samples poses.."+str(classes.numpy()))

        # generate the repr (joints3D/pose etc)
        model.eval()
        with torch.no_grad():
            noise_same_action = "random"
            noise_diff_action = "random"
            samples, labels, mask, lengths = dataset.get_label_sample_all(classes.numpy())
            if  samples.size()[0] > 60:
                maxclip = 60
            else:
                maxclip = samples.size()[0]
            gen = {"x": samples[:maxclip,:,:,:].to(model.device),
                   "y": labels.squeeze(0)[:maxclip].to(model.device),
                   "mask": mask.repeat(maxclip,1),
                   "lengths": lengths.to(model.device),
                   "output": samples[:maxclip,:,:,:].to(model.device)}
            model(gen)
            # Generate the new data
            generation = model.generate(gen["gamma"],gen["beta"],classes, gendurations, nspa=nspa,
                                        noise_same_action=noise_same_action,
                                        noise_diff_action=noise_diff_action,
                                        fact=fact)


            # x_rotations = generation["output"]
            #
            # x_rotations = x_rotations.permute(0, 3, 1, 2)
            # trans = x_rotations[:,:,-1:,:3]
            # x_rotations = x_rotations[:, :,:-1]
            # generation["output_rot"] = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(x_rotations[generation["mask"]]))

            #a = torch.cat((trans,generation["output_rot"].reshape(nspa, num_frames, *generation["output_rot"].shape[1:])),dim=2).cpu().numpy()
            a = generation["output_xyz"].permute(0,3,1,2).cpu().numpy()
            output.append(a)

    return output


def main():
    parameters, folder, checkpointname, epoch = parser()
    nspa = parameters["num_samples_per_action"]

    # no dataset needed
    if parameters["mode"] in []:   # ["gen", "duration", "interpolate"]:
        model = get_model(parameters)
    else:
        model, datasets = get_model_and_data(parameters)
        dataset = datasets["train"]  # same for ntu

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)

    from src.utils.fixseed import fixseed  # noqa
    for seed in [1]:  # [0, 1, 2]:
        fixseed(seed)
        # visualize_params
        onlygen = True
        vertstrans = False
        inter = True and onlygen
        varying_beta = False
        if varying_beta:
            betas = [-2, -1, 0, 1, 2]
        else:
            betas = [0]
        for beta in betas:
            output = generate_actions(beta, model, dataset, epoch, parameters,
                                      folder, inter=inter, vertstrans=vertstrans,
                                      nspa=nspa, onlygen=onlygen)
            if varying_beta:
                filename = "generation_beta_{}.npy".format(beta)
            else:
                filename = "generation.npy"
            output = np.array(output)
            output = np.concatenate(output)
            filename = os.path.join(folder, filename)
            np.save(filename, output)
            print("Saved at: " + filename)


if __name__ == '__main__':
    main()
