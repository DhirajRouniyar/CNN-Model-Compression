# CNN-Model Compression
Aim is to reduce the model size, run-time memory, computing operations, while introducing no accuracy loss to and minimum overhead to the training process.  

Reference Paper: "[Learning Efficient Convolutional Networks Through Network Slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) (ICCV2017)." .

##Results  

|  CIFAR10-VGG16BN  | Baseline | Trained with Sparsity (1e-4) | Pruned (0.7 Pruned) | Fine-tuned (40epochs) |
| :---------------: | :------: | :--------------------------: | :-----------------: | :-------------------: |
| Top1 Accuracy (%) |  93.62   |            93.77             |        10.00        |         93.56         |
|    Parameters     |  20.04M  |            20.04M            |        2.42M        |         2.42M         |

|             Pruned Ratio             |       0       |     0.1      |      0.2      |     0.7      |
| :----------------------------------: | :-----------: | :----------: | :-----------: | :----------: |
| Top1 Accuracy (%) without Fine-tuned |     93.77     |    93.72     |     93.76     |    10.00     |
|       Parameters(M) / macc(M)        | 20.04/ 398.44 | 15.9/ 349.22 | 12.28/ 307.78 | 2.42/ 210.84 |


## Train with Sparsity

```shell
python main.py -sr --s 0.0001
```

## Pruning

```shell
python prune.py --model model_best.pth.tar --save pruned.pth.tar --percent 0.7
```

## Fine-tuning

```shell
python main.py -refine pruned.pth.tar --epochs 40
```

