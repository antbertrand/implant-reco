# Action Plan

This will describe how we plan on achieving the best performances, what are the current results, what are the next steps and all the decisions we take on those matters.

## Dataset used

### Using the balanced
Epoch 24/50
2641 - 164s - loss: 0.1011 - regression_loss: 0.0926 - classification_loss: 0.0085 - val_loss: 0.1172 - val_regression_loss: 0.1064 - val_classification_loss: 0.0108

Parsing annotations: 100% (200 of 200) |#| Elapsed Time: 0:00:00 Time:  0:00:00
86 instances of class 0 with average precision: 1.0000
91 instances of class 1 with average precision: 1.0000
68 instances of class 2 with average precision: 1.0000
82 instances of class 3 with average precision: 1.0000
92 instances of class 4 with average precision: 1.0000
88 instances of class 5 with average precision: 1.0000
98 instances of class 6 with average precision: 1.0000
82 instances of class 7 with average precision: 1.0000
81 instances of class 8 with average precision: 1.0000
103 instances of class 9 with average precision: 1.0000
301 instances of class / with average precision: 1.0000
96 instances of class A with average precision: 1.0000
114 instances of class B with average precision: 1.0000
102 instances of class C with average precision: 0.9890
89 instances of class D with average precision: 1.0000
92 instances of class E with average precision: 1.0000
97 instances of class F with average precision: 0.9794
94 instances of class G with average precision: 0.9996
98 instances of class H with average precision: 1.0000
95 instances of class I with average precision: 1.0000
94 instances of class J with average precision: 1.0000
93 instances of class K with average precision: 1.0000
82 instances of class L with average precision: 1.0000
76 instances of class M with average precision: 1.0000
94 instances of class N with average precision: 1.0000
96 instances of class O with average precision: 0.9985
103 instances of class P with average precision: 0.9743
85 instances of class Q with average precision: 0.9995
98 instances of class R with average precision: 1.0000
95 instances of class S with average precision: 1.0000
95 instances of class T with average precision: 1.0000
95 instances of class U with average precision: 1.0000
84 instances of class V with average precision: 1.0000
96 instances of class W with average precision: 1.0000
95 instances of class X with average precision: 1.0000
78 instances of class Y with average precision: 1.0000
92 instances of class Z with average precision: 1.0000
mAP: 0.9983



#### DL the dataset

The dataset is stored on Azure, the URL of the blob container is the following :
*https://eurosilicone.blob.core.windows.net/dsdetection*

This will give you a dataset split in three : a train, validation and test set. The split has been done manually, to make sure very similar images do not end up in different splits, which would distort the results.
