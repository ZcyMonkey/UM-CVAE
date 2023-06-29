# UM-CVAE

Code of ECCV 2022 paper: Learning Uncoupled-Modulation CVAE for 3D Action-Conditioned Human Motion Synthesis

Abstract:Motion capture data has been largely needed in the movie and game industry in recent years. Since the motion capture system is expensive and requires manual post-processing, motion synthesis is a plausible solution to acquire more motion data. However, generating the action-conditioned, realistic, and diverse 3D human motions given the semantic action labels is still challenging because the mapping from semantic labels to real motion sequences is hard to depict. Previous work made some positive attempts like appending label tokens to pose encoding and performing action bias on latent space. However, how to synthesize diverse motions that accurately match the given label is still not fully explored. In this paper, we propose the Uncoupled-Modulation Conditional Variational AutoEncoder(UM-CVAE) to generate action-conditioned motions from scratch in an uncoupled manner. The main idea is twofold: (i)training an action-agnostic encoder to weaken the action-related information to learn the easy-modulated latent representation; (ii)strengthening the action-conditioned process with FiLM-based action-aware modulation. We conduct extensive experiments on the HumanAct12, UESTC, and BABEL datasets, demonstrating that our method achieves state-of-the-art performance both qualitatively and quantitatively with potential applications.

# Acknowledgements

Part of the code is borrowed from "Action2Motion: Conditioned Generation of 3D Human Motions" and " Action-conditioned 3D human motion synthesis with Transformer VAE"
