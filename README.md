## Apply Attention based Seq2seq OCR Recognition Model

This code implements recognition part of the ASTER 

The text recognizer is an sequence to sequence model based on attention. The origin paper can be found at [Here](https://ieeexplore.ieee.org/abstract/document/8395027)

#### Train

```
python3 train.py [train_res_path] [valid_res_path] [model_path]
```

#### Test
```
python3 test.py [test_res_path]  [model_path]
```



#### Results

I trained the model based on Invoice dataset which contain 1830 vocabularies and 18710 training images and 6752 validation images. The final accuracy on the validation data is about 70.2% on line base accuracy and 90.1% on word base accuracy.  

