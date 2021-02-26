# gestures-detection-model
Our model helps us to detect the gestures of participants in a meeting room, suddenly, we try to detect the ``` voting ```, ``` speaking``` gestures of a ```person``` participant.
----

![image_exp](https://github.com/linto-ai/gestures-detection-model/blob/main/demo/gest.jpg)

----


This model of gesture detection is inspired by the project [Single Shot MultiBox Detector Implementation in Pytorch](https://github.com/qfgaohao/pytorch-ssd)

We base our gesture detection work on the [VGG16-SSD model](https://storage.googleapis.com/models-hao/vgg16-ssd-mp-0_7726.pth) and by [transfer learning](https://pdf.sciencedirectassets.com/271526/1-s2.0-S0262885619X00128/1-s2.0-S0262885619304469/am.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjENj%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIDG5rgcNCSaknHOga5r4gN%2FjYKAPdHTQKRVOJvZjOVdIAiEA9d2el3EgsAmF1%2FuLEjTBUicrL098OI%2FlPX%2FzzI0pNn8qvQMIwf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARADGgwwNTkwMDM1NDY4NjUiDNos7Fcg6UkxaL5EHCqRAxymTIOeo5BJILFkY2JSQYf78X14L9TViLZ6GWuFa7gT0N7DLUFc2DJJUC7ze4FjXIHnSG7NVoS9dTrHOCtmr7%2BZ9C%2Fo9FRauKtGx2MDei4lWYD3%2FUIzarKjMyvxqTwVmWCkzI7Ym9H1HlDqNK0e93QLDopyEtDTeW7AnU8nJgaGomQx5NqeKlYSYCPkmbITwW%2F7HOeeF0gWb8SIs5PpRtKqxiWB2CVBIEkBsVF5jhiWWavHZgdMlH%2BTSquPy2s3pmUTZDcrgkzEFqr6hGyuuoiBvhcwticcnDCva%2B%2Fuk6vu2sy0HqbMzq%2Fxzc3dAxS5X%2FrNJUmRCSu4C%2FkUN8p4lQtbByNqcXsMN4j8RyvhgjZYsfb1nmjTO3a1oCp53nSBoeMJPzrEHntquNaHWvAzpd4c%2BHTrMF110YlJInLYQ4gxaaizpNkozHAkeaYeALAOVU0DeGxe19sb3%2Fr8Gk2FERzH%2FHh%2FLnTljUPOis%2F1VRJUgZnRUFfuMnfMgZlOjzgnPKEeUzpmKyJmWEtpW0ewAf24MMLK4IAGOusB12cKGvcf3U5eft4ONG2qo6Dv8GjqYXoBJDRCJ3kkDnu6S89G3Lm37XCnCkxx7ta2PwyPl2SSRMLbTS4R61mIVeY4BpUvGn1vWSPiivhJkLkMgp%2B0PJe1cZ%2BrSZb9INdfWEDMVfW6gaJN0nU1TU1xalPzR%2BqGiVtLTWj2n7F0XWiqPev0DZ%2FTCEOEsySWhs0Bwmg7RCNtAD4%2FSr%2Fkqt3MLX4AnpTK5vtZ4m6hjYvcVMNt4FHxzVmXTMW2rKZATnLqlrJqHl2A2VO3rO%2BlqVxqEabzB3dxl3m2iCiAazx9NUTihFOFT4ZUNWO2Ew%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20210201T170148Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYVSWEVZJD%2F20210201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c0d02a0f527d7c5073181cd25ec6c1c6d7a13b05813a40197c220f393ce28c30&hash=66869a12eb99d346cfd64e0acbc3f7aff2ec03cd3c26b956a9c764f68500d916&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0262885619304469&tid=pdf-38918c10-7411-4466-b3fc-c9248d4ed51a&sid=6f5b38eb8c100948685ba5512e0ee051d219gxrqb&type=client) on the panoramic image [data](https://github.com/linto-ai/panoramic-dataset-for-gestures-detection) of [LINAGORA](https://linagora.com/)
---
### To train and test our model, we need to have all the dependencies:
1. Python 3.6+
2. Git, Wget 
3. OpenCV
4. Pytorch 1.0 or Pytorch 0.4+
5. Pip3 or Pip
6. Numpy
7. Pandas

---

### Run the following commands to ``` train ``` our model

1. Clone from github the Pytorch-SSD repository
```
git clone https://github.com/qfgaohao/pytorch-ssd

cd pytorch-ssd
```
2. Install requirements
```
pip3 install -r models/requirements.txt
```
3. Download dataset  ( panoramic images and videos )
```
git clone https://github.com/linto-ai/panoramic-dataset-for-gestures-detection

mv panoramic-dataset-for-gestures-detection data
```
4. Download Vgg16-SSD basic model
```
wget -P models https://storage.googleapis.com/models-hao/vgg16-ssd-mp-0_7726.pth
```
5. Download model and the label file ( voc-model-labels.txt )
```
git clone https://github.com/linto-ai/gestures-detection-model

cp gestures-detection-model/voc-model-labels.txt models/
```
6. Train VGG16-SSD model with transfer learning for 100 epochs
```
python3 train_ssd.py --dataset_type voc --datasets data --net vgg16-ssd --pretrained_ssd models/vgg16-ssd-mp-0_7726.pth --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 1 --num_epochs 100 --base_net_lr 0.001  --batch_size 5
```
Output:

![image for train](https://github.com/linto-ai/gestures-detection-model/blob/main/demo/train_vgg16.png)

### Run the following commands to ``` test ``` our model

1. Change the model path
```
cp -r gestures-detection-model/vgg16-ssd-linagora-gest-detection.pth models/
``` 

2. Run image demo
```
python3 run_ssd_example.py vgg16-ssd models/vgg16-ssd-linagora-gest-detection.pth models/voc-model-labels.txt test/img1.jpg
```
![image_demo](https://github.com/linto-ai/gestures-detection-model/blob/main/demo/linagora_test-gest1.jpg)
3. Run video demo 
```
python3 run_ssd_live_demo.py vgg16-ssd models/vgg16-ssd-linagora-gest-detection.pth models/voc-model-labels.txt test/vid1.avi
```
![video_demo](https://github.com/linto-ai/gestures-detection-model/blob/main/demo/gest.gif)
