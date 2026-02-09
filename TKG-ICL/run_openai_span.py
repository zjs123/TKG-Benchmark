import torch
from tqdm import tqdm

from openai_utils import predict_chatgpt_span
from utils import (
    MAEMetric,
    get_args,
    get_filename,
    load_data_span,
    prepare_input_span,
    update_metric_span,
)

if __name__ == "__main__":
    args = get_args()

    test_sample, background_space, ent_dict, rel_dict, entity_2_des = load_data_span(args)
    metric = MAEMetric()
    filename = get_filename(args)
    
    tmp_test_samples = []
    yes_num = 0
    test_timestamps = sorted(test_sample.keys())
    tmp_test_samples = test_sample[test_timestamps[0]]
    with torch.no_grad(), open(filename, "w", encoding="utf-8") as writer, tqdm(test_timestamps[1:]) as pbar:
        for i, timestamp in enumerate(pbar):
            remain_test_samples = []
            for test_fact in tmp_test_samples:
                model_input = prepare_input_span(test_fact, timestamp, background_space, ent_dict, rel_dict, entity_2_des, args, return_prompt=True)
                predictions = predict_chatgpt_span(model_input, args)
                if predictions == -1:
                    continue
                elif predictions == 1:
                    update_metric_span(test_fact, metric, timestamp, args)
                    yes_num += 1
                else:
                    remain_test_samples.append(test_fact)
            
            tmp_test_samples = remain_test_samples
            tmp_test_samples.extend(test_sample[timestamp])
            pbar.set_postfix(metric.dump())
    
        if len(remain_test_samples) != 0:
            for test_fact in remain_test_samples:
                update_metric_span(test_fact, metric, "*"+str(len(test_timestamps)), args)
    print(metric.dump())
    print(yes_num)
    
