from src.parser.evaluation import parser
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
def main():
    parameters, folder, checkpointname, epoch, niter = parser()

    dataset = parameters["dataset"]
    print(dataset)
    if dataset in ["ntu13", "humanact12"]:
        from src.evaluate.gru_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter)
    elif dataset in ["uestc"]:
        from src.evaluate.stgcn_eval import evaluate
        evaluate(parameters, folder, checkpointname, epoch, niter)
    else:
        raise NotImplementedError("This dataset is not supported.")



if __name__ == '__main__':
    main()
