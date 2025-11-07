# RCIMVC
@article{wu2025relationship,
  title={Relationship completion for incomplete multi-view clustering},
  author={Wu, Minghong and Zhu, Jihua and Yan, Wenbiao and Chen, Bin and Zheng, Qinghai},
  journal={Neural Networks},
  pages={107791},
  year={2025},
  publisher={Elsevier}
}

We welcome citations of our work.


## Environment

python == 3.9.13

pytorch== 1.13.0

Other requirements to be specified in requirements.txt.


## run 

Modify the flag in the train to select the data set, and then run the python train.py, train_3v.py or train_4v.py command.


## Datasets

Note that if you want to run your own dataset, you need to add the way to read the dataset in the datasets.py file and add network parameters in config.py. 

Then select the data set flag in the train.py and run again.


## Thanks
Our work is inspired by
@inproceedings{chao2024incomplete,
  title={Incomplete contrastive multi-view clustering with high-confidence guiding},
  author={Chao, Guoqing and Jiang, Yi and Chu, Dianhui},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={38},
  number={10},
  pages={11221--11229},
  year={2024}
}.

