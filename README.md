## WaveMsNet

**Task:** Environment sounds classification

**Abstract**—Deep learning has dramatically improved the perfor- mance of sounds recognition. However, learning acoustic models directly from the raw waveform is still challenging. Current waveform-based models generally use time-domain convolutional layers to extract features. The features extracted by single size filters are insufficient for building discriminative representation of audios. In this paper, we propose multi-scale convolution operation, which can get better audio representation by improving the frequency resolution and learning filters cross all frequency area. For leveraging the waveform-based features and spectrogram-based features in a single model, we introduce two-phase method to fuse the different features. Finally, we propose a novel end-to-end network called WaveMsNet based on the multi-scale convolution operation and two-phase method. On the environmental sounds classification datasets ESC-10 and ESC-50, the classification accuracies of our WaveMsNet achieve 93.75% and 79.10% respectively, which improve significantly from the previous methods.

![WaveMsNet](https://github.com/Black-Black-Man/WaveMsNet/blob/master/figures/WaveMsNet.png)

This work is submitted to IJCNN 2018, paper will be published soon.


## Files structure
	
	./
	├── LICENSE
	├── README.md
	├── cross_fold
	│   ├── evaluate-setup-ESC10
	│   │   ├── fold0_test.txt
	│   │   ├── ...
	│   │   └── fold4_valid.txt
	│   ├── evaluate-setup-ESC50
	│   │   ├── fold0_test.txt
	│   │   ├── ...
	│   │   └── fold4_valid.txt
	│   └── src
	│       ├── ESC10.audiolist
	│       ├── ESC50.audiolist
	│       └── make_files.py
	├── figures
	│   ├── WaveMsNet.png
	│   ├── comparison.png
	│   └── confusion_matrix.png
	└── src
	    ├── data_process.py
	    ├── data_transform.py
	    ├── id_lb.yaml
	    ├── main.py
	    ├── network.py
	    └── util.py
	    
## Requirments
Python 3.6

PyTorch 0.3.0

Torchvision 0.2.0

Librosa 0.5.1

## Data preparation

### Download datasets
Dataset for Environmental Sound Classification could be downloaded [here](https://github.com/karoldvl/ESC-50).

### Divide audios into 5-cross-folds
	cd cross_folds/src
	python make_file.py

**Note:** You should adjust code with different datasets (ESC-10 or ESC-50).

### Tranfer data

	cd ../../src
	python data_transform.py
	
Tranfer each audio clip into a dictionary which has three keys: label, key(filename), data. For eample:

`2-77945-A.ogg` will transfer to:

`{'lable': 8, 'key':'2-77945-A', 'data': array([-1.2207, -1.8310,...], dtype=float32)}`

Then, we store them with pickle format.

## Network training

	python main.py argument
	python main.py --mode=test --mode='...'
	
You will see the training process:

```
WaveMsNet
Epoch:1 (12.5s) lr:0.01  samples:1200  Loss:3.959  TrainAcc:3.50%
Epoch:2 (11.1s) lr:0.01  samples:1200  Loss:3.680  TrainAcc:6.17%
Epoch:3 (11.1s) lr:0.01  samples:1200  Loss:3.389  TrainAcc:10.17%
Epoch:4 (11.1s) lr:0.01  samples:1200  Loss:3.117  TrainAcc:16.17%
Epoch:5 (11.1s) lr:0.01  samples:1200  Loss:2.937  TrainAcc:19.08%
...
Test set: Average loss: 17.967 (18.9s), TestACC: 260/400 65.00%

model has been saved as: ../model/WaveMsNet_fold0_epoch80.pkl
...
```
Parameters could be changed. For example: *batch_size, epochs, learning_rate, momentum, network, ...*

## Result analysis

### Other network

![Compare with other network](https://github.com/Black-Black-Man/WaveMsNet/blob/master/figures/comparison.png)

We employ different backend networks, all of which are widely used and well-preformed in the field of image. They are AlexNet, VGG (11 layers with BN) and ResNet (50-layers). The multi-scale models consistently outperform single-scale models. It indicates that multi-scale models have a wide range of effectiveness. 
### Confusion matrix

![Confusion matrix](https://github.com/Black-Black-Man/WaveMsNet/blob/master/figures/confusion_matrix.png)

ESC-50 is more challenge than ESC-10 dataset, we report the confusion matrix across all folds on ESC-50. The results suggest our approach obtains very good performance on most categories, such as baby crying (95% accuracy) or clock alarm (97% accuracy). Common confusions are helicopter confused as airplane, vacuum cleaner confused as train. Actually, these sounds are also challenge for human to distinguish.
