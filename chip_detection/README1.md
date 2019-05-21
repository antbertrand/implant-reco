
**NOTE:** 
- 
- 
- 

# choice of the model architecture for 3 class classification

- Pretrained vgg16 :    	Acc : 96,6%	
				[[27  0  0]
			 	[ 0 57  2]
			 	[ 0  3 56]]
				
				Inference time = 0.6 sec

- Simple CNN : 			Acc: 91.7%
				[[27  0  0]
				 [ 1 50  8]
				 [ 0  3 56] 

				Inference time = 0.074

- Pre-trained inceptionv3	Acc: 92.4%
				[[27  0  0]
				 [ 0 57  2]
				 [ 0  9 50]]

				Inference time = 






## Performances on ImageNet

Networks                            | AlexNet     |     VGG16   |     VGG19   |
-------------------------------------------------------------------------------
Top 1 Error                         |   42,94%    |   32,93%    |   32,77%    |
Top 5 error                         |   20,09%    |   12,39%    |   12,17%    |
Top 10 error                        |   13,84%    |    7,77%    |    7,80%    |
Number of params                    |     61M     |     138M    |     144M    |
Prediction time, batch of 64 (GPU)  |   0.4101s   |   0.9645s   |   1.0370s   |
Prediction time, single image (CPU) |   0.6773s   |   1.3353s   |   1.5722s   |
```



## Useful functions for ImageNet


#### Converting synsets to ids


#### Getting all the children of a synset 


## Credits
* For the AlexNet network, we have adapted the weights that can be found here : 
Taylor, Graham; Ding, Weiguang, 2015-03, <i>"Theano-based large-scale visual recognition with multiple GPUs"</i>, <a href="http://hdl.handle.net/10864/10911">hdl:10864/10911</a> University of Guelph Research Data Repository 

* For the VGG networks, we have adapted the code released by baraldilorenzo here : https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
We changed it to have the "heatmap" option, and we modified the weights in the same way.
