# [Towards Better Evolution Modeling for Temporal Knowledge Graphs (Submitted to KDD 2026)]()

## Dataset
All four temporal knowledge graph datasets proposed in this paper can be downloaded from [here](https://drive.google.com/file/d/12DE8HfA5JogHcsrXi4oPxIIsAcX7cjlR/view?usp=sharing).

<img width="1230" alt="image" src="https://github.com/zjs123/TKG-Benchmark/blob/main/dataset_statics.png">

### Data Format
Each dataset is preserved as a pickle file, which consists of 6 items.
* __train_split_time:__ stores the final timestamp of the training set.
* __valid_split_time:__ stores the final timestamp of the validation set.
* __train_set:__ stores the knowledge for training, which consists of timestamp knowledge in the form of (s,r,o,t), and time-interval knowledge in the form of (s,r,o,t_s, t_e).
* __valid_set:__ stores the knowledge for validation, which consists of timestamp knowledge in the form of (s,r,o,t), and time-interval knowledge in the form of (s,r,o,t_s, t_e).
* __test_set:__ stores the knowledge for test, which consists of timestamp knowledge in the form of (s,r,o,t), and time-interval knowledge in the form of (s,r,o,t_s, t_e).
* __e_2_des_dict:__ stores the mapping from the original entity name to its short textual descriptions.

After downloading the datasets, they should be uncompressed into the `datasets` folder. The performance of the co-occurrence-based scoring strategy can be obtained by running the `dataset_analysis.ipynb` file.

## Reproduce the Results

### Generative Knowledge Forecasting Task
* Example of training *TNT* on *FinWiki* dataset:
```{bash}
cd framework
python tkge.py train --config config/example_tnt_FinWiki_stamp.yaml
```

* Example of run *TKG-ICL* on *FinWiki* dataset:
```{bash}
cd TKG-ICL
python run_openai.py --dataset FinWiki
```
* The metrics on the test set  will be automatically saved in `results/TNT/FinWiki/ex0000/logging/` folder
* The best checkpoint will be saved in `results/TNT/FinWiki/ex0000/ckpt/` folder, and the checkpoint will be used to reproduce the performance.

### Knowledge Obsolescence Prediction Task
* Example of training *TNT* on *FinWiki* dataset:
```{bash}
cd framework
python tkge.py train --config config/example_tnt_FinWiki_span.yaml
```

* Example of run *TKG-ICL* on *FinWiki* dataset:
```{bash}
cd TKG-ICL
python run_openai_span.py --dataset FinWiki
```
* The metrics on the test set  will be automatically saved in `results/TNT/FinWiki/ex0000/logging/` folder
* The best checkpoint will be saved in `results/TNT/FinWiki/ex0000/ckpt/` folder, and the checkpoint will be used to reproduce the performance.

## Contact
For any questions or suggestions, you can use the issues section or contact us at (zjss12358@gmail.com).

## Acknowledge
Codes and model implementations are referred to [TKG-ICL](https://github.com/usc-isi-i2/isi-tkg-icl) project and [Unified_Framework_of_Temporal_Knowledge_Graph_Models](https://github.com/TemporalKGTeam/A_Unified_Framework_of_Temporal_Knowledge_Graph_Models) project. Thanks for their great contributions!

### Reference
```
@article{zhang2026tkgbench,
  title={Towards Better Evolution Modeling for Temporal Knowledge Graphs},
  author={Zhang, Jiasheng and Li, Zhengpin and Wang, Mingzhe and Shao, Jie and Cui, Jiangtao and Li, Hui},
  journal={},
  year={2026}
}
```

