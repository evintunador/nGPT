# nGPT
My best attempt at replicating/teaching Nvidia's Normalized-GPT. The vast majority of it (see first commit from Nov 18, 2024) was written only from reading the paper and without awareness of the fact that Nvidia had open-sourced [their own implementation](https://github.com/NVIDIA/ngpt/blob/main/model.py) (see also [lucidrains' implementation](https://github.com/lucidrains/nGPT-pytorch)). Click below to watch the video:

[![ERROR DISPLAYING IMAGE, CLICK HERE FOR VIDEO](https://img.youtube.com/vi/s5XXYa5aeQU/0.jpg)](https://www.youtube.com/watch?v=s5XXYa5aeQU)

# getting started
## setting up
1. clone the repo, create your venv, and install everything in requirements.txt
## learning
1. read thru `tutorial.ipynb`, watch [the video](https://www.youtube.com/watch?v=s5XXYa5aeQU) and read [the paper](https://arxiv.org/abs/2410.01131v1)

# using
## training
1. edit values in `config.py`
2. run all cells in `train.ipynb`
## inference
1. open `inference.ipynb` 
2. set the variable `model_name` equal to the name of a sub-folder within `models/`. defaults to the tiny model I trained "nGPT-2m"
3. run the rest of the cells. Notice that temperature needs to be very weirdly small in order to get reasonable output; this has something to do with the fact that we're using cosine similarity and the scaling factor. Not sure if that quirk would persist in a model of proper size