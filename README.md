### Overview

The code is repository for ["DEFEAT: Decoupled Feature Attack Across Deep Neural Networks"](https://www.sciencedirect.com/science/article/pii/S0893608022003434) (Neural Networks).

### Prerequisites

python **3.6**  
tensorflow **1.14**  

### Pipeline 
<img src="/figure/overview.png" width = "700" height = "250" align=center/>

### Run the Code  
Run DEFEAT to generate adversarial examples: `main.py`.  

### Experimental Results
We attack four normally trained models to generate adversarial examples, and test the transferability against defense models.

<b>Layer Transferability</b>

<img src="/figure/exp1.png" width = "700" height = "350" align=center/>

<b>Standalone Experiment</b>

<img src="/figure/exp3.png" width = "600" height = "250" align=center/>

<b>Ensemble Experiment</b>

<img src="/figure/exp4.png" width = "700" height = "250" align=center/>


### Citation
If you find this project is useful for your research, please consider citing:

	@article{huang2022defeat,
	  title={DEFEAT: Decoupled feature attack across deep neural networks},
	  author={Huang, Lifeng and Gao, Chengying and Liu, Ning},
	  journal={Neural Networks},
	  volume={156},
	  pages={13--28},
	  year={2022},
	  publisher={Elsevier}
	}



