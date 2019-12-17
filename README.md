# Python NN_Template and Project Hand Writing Recognizing (HWR)

### Project

- **NN_Template** : A python class for universe NN project

- **HWR**: read a picture of any resolution, then give which number it is

  >  Test by mnist **train.csv** and **test.csv**, when set epochs=7 and rotate_expand=5 Pi/180, it reachs 97% for accuracy.
- **Back_Transport**: We set the result and reversely to generate the img, which may provide new sight of NN.
![](http://kylinhub.oss-cn-shanghai.aliyuncs.com/2019-05-01-back_1_1_7-1.png)
- **Epoch_Trend**: To find which which epoch can help us get the maximal and most stable accuracy, which may give idea in batch_size choice in pytorch.
- **Rotate_trend**: For 60000 pictures(28 * 28 * 1 matrix), we can use rotate expand to get more training data to get more correctness.
<center>
<img src="http://kylinhub.oss-cn-shanghai.aliyuncs.com/2019-05-03-epoch_trend_DR_7_lr051.png" width="50%" height="50%" />
</center>

