The code is based on
[**Automatic Data Augmentation for Generalization in Deep Reinforcement Learning**](https://arxiv.org/pdf/2006.12862.pdf) by 


# Requirements
The code was run on a GPU with CUDA 10.2.
To install all the required dependencies: 

```
conda create -n auto-drac python=3.7
conda activate auto-drac

git clone git@github.com:rraileanu/auto-drac.git
cd auto-drac
pip install -r requirements.txt

git clone https://github.com/openai/baselines.git
cd baselines 
python setup.py install 

pip install procgen
```


# Instructions
```
cd auto-drac
```

replace DrAC model.py in ucb_rl_meta with provided model.py.
Add Siet.py to ucb_rl_meta folder
replace drac.py in  ucb_rl_meta/algo with provided drac.py.
add tain2.py to auto-drac folder.
NOTE: the model.py and SieT.py are the only required ones. you may also use the default procgen codebase. 
We added --device_id to submit to specific GPUs.



## Train SieT with DrAC with *crop* augmentation on StarPilot
```
python train2.py --env_name starpilot --device_id 0 --seed 1 --use_sit True --choice 0  --run_name Siet --num_mini_batch 96  --ppo_epoch 2  --hidden_size 64   
```

