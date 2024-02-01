import pdb
import torch
from tqdm import tqdm
from utils.logger import LOGGER, TQDM_BAR_FORMAT

def evaluate(model, dataloader, metric):
    pbar = enumerate(dataloader)
    pbar = tqdm(pbar, total=len(dataloader), bar_format=TQDM_BAR_FORMAT)

    model.eval()
    for idx, data_batch in pbar:
        with torch.no_grad():
            if type(model) in [torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel]:
                data = model.module.data_preprocessor(data = data_batch, training = False)
            else:
                data = model.data_preprocessor(data = data_batch, training = False)
            outputs = model(**data, mode = 'predict')
            metric.process(data_batch = data_batch, data_samples = outputs)

            pbar.set_description('Evaluating on validation dataset')

    results = metric.evaluate(len(dataloader.dataset)) # 每次执行完evaluate后会自动清空结果
    return results
