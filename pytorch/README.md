# Monocular Depth Estimation With Semantic Information And Multiple Contraints 
We propose our monocular depth estimation model that leverages high-level and multi- scale information and dynamically adjusts the field of view to achieve state-of-the-art performance. Finally, we apply multi- loss to limit the development of features and ensure accuracy after fusion.

![](https://i.imgur.com/9rg3G1P.jpg)
## Demo
KITTI
![](https://i.imgur.com/fexqX86.jpg)

NYUv2
![](https://i.imgur.com/o1ZW9Wp.jpg)


## Dockerfile
使用上面提供的 dockerfile
``` bash
$ sudo docker build . -t 名字
```
或是從 docker hub 拉下來
``` bash
$ sudo docker pull xiaosean/mvclabsharepytorch
```
執行時若有缺少套件，可用 pip 自行補上
## Prepare KITTI Dataset
下載 [data_depth_annotated.zip](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip) 並且全部解壓縮，裡面有 KITTI 全部圖片的深度 ground truth
接著用底下的指令下載 kitti 的訓練資料
``` bash
$ aria2c -x 16 -i ../utils/kitti_archives_to_download.txt
$ parallel unzip ::: *.zip
```
全部解壓縮之後會有五個資料夾，會長的下底下這樣
```
kitti_dataset
   |—————— 2011_09_26
   |       
   |—————— 2011_09_28
   |        
   └—————— 2011_09_29
   |
   └—————— 2011_09_30
   |
   └—————— 2011_10_03
   |
   └—————— data_depth_annotated

```
## Prepare NYUV2 Dataset
透過底下的指令下載 NYUV2 的 training set 和 testing set
``` bash
$ cd ~/workspace/utils
### Get official NYU Depth V2 split file
$ wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
### Convert mat file to image files
$ python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../dataset/nyu_depth_v2/official_splits/
```
解壓縮後，所有 dataset 的資料夾配置如下
```
dataset
   |—————— kitti_dataset
   |        └—————— 2011_09_26
   |        └—————— 2011_09_28
   |        └—————— 2011_09_29
   |        └—————— 2011_09_30
   |        └—————— 2011_10_03
   |        └—————— data_depth_annotated
   |
   |       
   |—————— nyu_depth_v2
   |        └—————— official_splits
   |            └—————— test
   |            └—————— train
   |————————————————————————————————————
```
## Semantic Segmentation Weights
### KITTI
Semantic Segmentation 採用 [Panoptic-DeepLab (CVPR 2020)](https://github.com/bowenc0221/panoptic-deeplab) 這篇論文，其中的 pretrain 在[這裡](https://github.com/bowenc0221/panoptic-deeplab/blob/master/tools/docs/MODEL_ZOO.md)下載，把下載好的權重放入 ./pytorch/weights 中，並且依據現在模型的 backbone 導入對應的權重(ResNet101,ResNet50)，導入權重路徑可在 bts.py 中的 model_state_file 進行修改，另外在 arguments_train_eigen.txt 中的 --cfg configs/ 也要改成對應的 .yaml 檔，--encoder 也要改成對應的 backbone
### NYUv2
Semantic Segmentation 採用 [ShapeConv](https://github.com/hanchaoleng/shapeconv) 這篇論文，其中的 pretrain 在[這裡](https://github.com/hanchaoleng/ShapeConv/blob/master/model_zoo/README.md)下載，把下載好的權重放入 ./pytorch/weights 中，並且依據現在模型的 backbone 導入對應的權重(ResNet101,ResNet50)，導入權重路徑可在 bts.py 中的 model_state_file 進行修改，另外在 arguments_train_eigen.txt 中的--encoder 也要改成對應的 backbone
## Weights and Biases
訓練時有導入 [wandb](https://wandb.ai/site) 來查看訓練狀態，請自行事先下載，在 bts_main.py 搜尋 wandb，就可以找到相關設定

## Training on KITTI Dataset
### bts.py 的改動
訓練在 KITTI Dataset 時，為了接上 segmentation 的 pretrain weights， 在大約 400 行的 class BtsModel 請改成以下這種樣式，model_state_file 請自行修改，若有疑問可以查看範例檔案 **bts_kitti.py**：
``` python
class BtsModel(nn.Module):
    def __init__(self, params):
        super(BtsModel, self).__init__()
        self.encoder = encoder(params)
        

        self.segmentation = build_segmentation_model_from_cfg(config)
        print('load seg')     
        model_state_file = './weights/panoptic_deeplab_R101_os32_cityscapes.pth' 
        if os.path.isfile(model_state_file):
            model_weights = torch.load(model_state_file)
            if 'state_dict' in model_weights.keys():
                model_weights = model_weights['state_dict']
                logger.info('Evaluating a intermediate checkpoint.')
            self.segmentation.load_state_dict(model_weights, strict=True)
            del self.segmentation.backbone
            del self.segmentation.decoder.instance_decoder
            del self.segmentation.decoder.instance_head
            print('log segModel success!!!')
            # logger.info('Test model loaded from {}'.format(model_state_file))
        else:
            if not config.DEBUG.DEBUG:
                raise ValueError('Cannot find test model.')
        self.decoder = bts(params, self.encoder.feat_out_channels, self.segmentation, params.bts_size)

    def forward(self, x, focal):
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat, focal)
```
訓練主要仰賴兩個檔案，一個是 bts_main.py，主要訓練的過程都會寫在這個檔案裡面，另一個是 arguments_train_eigen.txt，裡面有模型訓練時需要調整的參數，像是 dataset 的目錄在哪裡，batch size 設定多少，幾個 epoch 等等，同時也要依照上面 Semantic Segmentation Weights 的說明調整 pretrain weight 的檔案路徑。開始訓練的指令如下：
``` bash
$ python bts_main.py arguments_train_eigen.txt
```
訓練好的模型會存在 models/--model_name，這個資料夾中
## Training on NYUv2 Dataset
### bts.py 的改動
訓練在 NYUv2 Dataset 時，為了接上 segmentation 的 pretrain weights， 在大約 400 行的 class BtsModel 請改成以下這種樣式，model_state_file 和 Config.fromfile 請自行修改，若有疑問可以查看範例檔案 **bts_nyu.py**：

``` python
class BtsModel(nn.Module):
    def __init__(self, params):
        super(BtsModel, self).__init__()
        self.encoder = encoder(params)
        
        from rgbd_seg.models import build_model
        from rgbd_seg.models.utils.shape_conv import ShapeConv2d
        from rgbd_seg.utils import load_checkpoint
        from rgbd_seg.utils import Config
        from collections import OrderedDict
        cfg = Config.fromfile('./configs/nyu/nyu40_deeplabv3plus_resnext101_shape.py')
        self.segmentation = build_model(cfg['inference']['model'])
        print('load seg')        
        model_state_file = './weights/nyu40_deeplabv3plus_resnext101_shape.pth'
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file)
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(filename))
            if hasattr(self.segmentation, 'module'):
                print('load module')
                self.segmentation.module.load_state_dict(state_dict, strict=False)
            else:
                self.segmentation.load_state_dict(state_dict, strict=False)
                print('load model')
           
        self.decoder = bts(params, self.encoder.feat_out_channels, self.segmentation, params.bts_size)

    def forward(self, x, focal):        
        skip_feat = self.encoder(x)
        return self.decoder(skip_feat, focal)
```
大約在 232行的 outputs 那一段也要改成以下這樣
``` python
outputs = {}
outputs['c1'] = features[1]
outputs['c2'] = features[2]
outputs['c3'] = features[3]
outputs['c4'] = features[4]
outputs['c5'] = features[5]
```
跟在 kitti 上訓練一樣，只是把要丟入的 .txt 檔改成 nyu 對應的 .txt 檔，記得在 bts.py 中的 model_state_file 要改成 nyu 對應的 pretrain weight，arguments_train_nyu.txt 中--encoder 也要改成對應的 backbone。開始訓練的指令如下：
``` bash
$ python bts_main.py arguments_train_nyu.txt
```
## Testing and Evaluation with KITTI and NTUv2

主要分成兩個步驟：
### 產生預測圖片
先使用 bts_test.py 產生模型預測的圖片，參數在後面的 arguments_test_eigen.txt 進行調整，其中模型的路徑使用訓練時儲存下來的 --checkpoint 路徑，--encoder 和 --model_name 也要與訓練時的設定一致，才跑得動
``` bash
$ python bts_test.py arguments_test_eigen.txt
```
**NYUv2**:
``` bash
$ python bts_test.py arguments_test_nyu.txt
```
跑完之後會出現一個 result_模型名稱 的資料夾，模型預測的圖片存在 ./raw 裡面

### 輸出預測分數
接著要用剛才預測的圖片跟 ground truth 進行比較，輸入指令如下
``` bash
$ python ../utils/eval_with_pngs.py --pred_path result_bts_eigen_res2net50_v1b_mask4addloss4connect/raw/ --gt_path ../../dataset/kitti_dataset/data_depth_annotated/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --do_kb_crop --garg_crop
```
--pred_path 輸入模型預測的圖片路徑，--gt_path 加入 gt 的路徑，--dataset 看要用哪一個 dataset，後面的設定符合 kitti 的設定

**NYUv2** 的 Evaluation 的指令如下：
``` bash
$ python ../utils/eval_with_pngs.py --pred_path result_bts_nyu_v2_pytorch_densenet161/raw/ --gt_path ../../dataset/nyu_depth_v2/official_splits/test/ --dataset nyu --min_depth_eval 1e-3 --max_depth_eval 10 --eigen_crop
```
接著就會跑出分數像下面這樣：
``` bash
GT files reading done
45 GT files missing
Computing errors
     d1,      d2,      d3,  AbsRel,   SqRel,    RMSE, RMSElog,   SILog,   log10
  0.955,   0.993,   0.998,   0.060,   0.249,   2.798,   0.096,   8.933,   0.027
Done.
```
KITTI 中有圖片遺失是正常的，NYUv2 則沒有遺失圖片



### KITTI
| Model           | cap   | Abs Rel | Sq Rel | RMSE  | RMSE log |
| --------------- | ----- | ------- | ------ | ----- | -------- |
| Ours-Res2Net50  | 0-80m | 0.056   | 0.181  | 2.291 | 0.086    |
| Ours-Res2Net101 | 0-80m | 0.054   | 0.180  | 2.256 | 0.084    |
| Ours-Res2Net101 | 0-50m | 0.052   | 0.136  | 1.670 | 0.079    |

### NYUv2

| Model | cap   | Abs Rel | RMSE  | log10 |
| ----- | ----- | ------- | ----- | ----- |
| Ours  | 0-10m | 0.104   | 0.375 | 0.045 |

## Inference
執行時須加入的參數如下：
--model_name (model name)
--encoder (maybe res2net50_v1b_bts)
--max_depth (10 for NYU, 80 for kitti)
--checkpoint_path (model path)
--dataset (kitti or nyuv2)

```bash
$ python bts_sequence.py --image_path <image_folder_path>
```
