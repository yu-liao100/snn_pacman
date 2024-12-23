Requires:
	python 3.5
	pytorch 0.4
Important files:
	DQN.py
	pacmanDQN_Agents.py 


To test the DQN network, launch:
	python3 pacman.py -p PacmanDQN -n 200 -x 100 -l smallGrid

To train the DQN network, launch:
	python3 pacman.py -p PacmanDQN -n 3000 -x 2900 -l smallGrid

Where:
	-n = number of episodes
	-x = episodes used for training (graphics = off)

Remarks:
	the game files had to be updated for python3 (print was not working)
	the model has already been trained and wins most of the time
	the model has been optimized, it requires less then 30 000 episodes to converge
	
To test training, change:
	model_trained = False
		in pacmanDQN_Agents.py (line 26)


I used the Pacman game engine provided by the UC Berkley Intro to AI project:
http://ai.berkeley.edu/reinforcement.html

I'd like to cite:
https://github.com/tychovdo/PacmanDQN
His implementation in Tensorflow helped me configure the Neural Network architecture.




## 跑SNN_DQN要修改的地方：

1. DQN网络换掉
2. 模型权重保存地址换掉
3. 跑出来的txt文件记录地址换掉
4. gpu显卡注意要不同
5. 保存的图片地址换掉
7。跑的轮次数量换掉
8. 跑的layer换掉

6. LIFSpike中后面要加.to(torch.float64)确保类型一致
### 待解决问题
1. 为何load模型测试的时候DQN会影响结果