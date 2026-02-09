import torch
from tqdm import tqdm

from openai_utils import predict_chatgpt
from utils import (
    GenerativeMetric,
    adjust_top_k,
    get_args,
    get_filename,
    load_data,
    prepare_input,
    update_history,
    update_metric,
    write_results,
)

if __name__ == "__main__":
    args = get_args()

    test_data, head_search_space, background_space, ent_dict, rel_dict, entity_2_des, t_s_ro = load_data(args)

    adjust_top_k(test_data, args)

    metric = GenerativeMetric()
    filename = get_filename(args)
    tested_entity = set()

    with torch.no_grad(), open(filename, "w", encoding="utf-8") as writer, tqdm(test_data) as pbar:
        for i, x in enumerate(pbar):
            if i % args.world_size != args.rank:
                continue
            search_space = head_search_space
            if x[0] in tested_entity:
                continue
            tested_entity.add(x[0])
            model_input = prepare_input(x, search_space, background_space, ent_dict, rel_dict, entity_2_des, args, return_prompt=True)

            if args.model == "chatgpt":
                predictions = predict_chatgpt(model_input, args)
            else:
                predictions = predict(model_input, args)

            update_history(x, search_space, predictions, args)

            example = write_results(x, predictions, t_s_ro, writer, args)

            update_metric(example, metric, args)
            pbar.set_postfix(metric.dump())

            if metric.total >= 200: 
                break
    
