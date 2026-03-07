
# -*- coding: GB2312 -*-
import numpy as np  #数学运算库
import matplotlib.pyplot as plt #画图库

class BernoulliBandit:
    '''问题对象，多臂老虎机'''
    def __init__(self,K): # 在这里初始化各种参数
        self.probs=np.random.uniform(size=K) #生成一个随机数组，默认范围0~1，大小为K
        self.K=K
        self.best_prob_idx=np.argmax(self.probs) #返回最大值的下标
        self.best_prob=self.probs[self.best_prob_idx]

    def step(self,K): # 这里进行奖励，用二分法，随机数小于种子数则表示抽中返回1，否则返回0
        return 1 if np.random.rand()<self.probs[K] else 0
    

np.random.seed(1) #随机选取一个种子数
K=10
B=BernoulliBandit(K) #初始化一个机器对象
print("K=%d"%K)

class Solver:
    '''多臂老虎机算法基本框架(只是一个模板)'''
    def __init__(self,bandit): #初始化各种参数
        self.bandit=bandit
        self.counts=np.zeros(self.bandit.K) # 每个杆的尝试次数
        self.regret=0
        self.action=[] #动作数组，记录选择杆的序号
        self.regrets=[] #每一步之后的累计懊悔值,可以用于画图

        
    def update_regret(self,k):
        #计算累计懊悔并保存
        self.regret+=self.bandit.best_prob-self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run(self,run_nums):
        # 运行算法，run_nums表示总次数
        for _ in range(run_nums):
            k=self.run_one_step() #选择第k个
            self.counts[k]+=1 #对应摇杆尝试次数+1
            self.update_regret(k) #更新懊悔值
            self.action.append(k) #记录动作

        
    def run_one_step(self):
        # 进行一步运行，即摇动一个摇杆,具体不写在这里，在下面的算法中实现
        pass

def plot_results(solvers,solver_names): #可视化几种算法的懊悔值
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

'''以下展示三种具体的算法'''
class EpsilonGreedy(Solver):
    '''
    epsilon贪心算法
    每次以概率1-ε选择以往经验中期望奖励估值最大的那根拉杆（利用），以概率ε随机选择一根拉杆（探索）
    '''
    def __init__(self, bandit,epsilon=0.01,init_prob=1.0):
        super(EpsilonGreedy,self).__init__(bandit)# python 类的继承
        self.epsilon=epsilon #可以认为是学习率
        self.estimates=np.array([init_prob]*self.bandit.K) #初始化每个摇杆的概率为1

    def run_one_step(self): # 在具体算法类中重新定义覆盖父类的函数
        if np.random.random()<self.epsilon: #选择探索
            k=np.random.randint(0,self.bandit.K)
        else:
            k=np.argmax(self.estimates) #否则选择当前已知的概率最高的一个
        r=self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

    def run(self,run_nums): # 在具体算法类中重新定义覆盖父类的函数
        for _ in range(run_nums):
            k=self.run_one_step()
            self.counts[k]+=1
            self.update_regret(k)
            self.action.append(k)

np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(B, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit,init_prob=1.0):
        super(DecayingEpsilonGreedy,self).__init__(bandit)
        self.estimate=np.array([init_prob]*self.bandit.K)
        self.total_count=0 #总运行次数

    def run_one_step(self):
        self.total_count+=1
        if np.random.random()<1/self.total_count:
            k=np.random.randint(0,self.bandit.K)
        else:
            k=np.argmax(self.estimate)

        r=self.bandit.step(k)
        self.estimate[k]+=1./(self.counts[k]+1)*(r-self.estimate[k])
        return k

    def run(self,run_nums):
        for _ in range(run_nums):
            k=self.run_one_step()
            self.counts[k]+=1
            self.update_regret(k)
            self.action.append(k)

np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(B)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

class UCB(Solver):
    def __init__(self, bandit,coef,init_prob=1.0):
        super().__init__(bandit)
        self.estimate=np.array([init_prob]*self.bandit.K)
        self.coef=coef #控制不确定性比重U(a)的系数
        self.total_count=0
    
    def run_one_step(self):
        self.total_count+=1
        ucb=self.estimate+self.coef*np.sqrt(np.log(self.total_count)/(2*(self.counts+1))) #用公式计算奖励期望上置信界数组
        k=np.argmax(ucb)
        r=self.bandit.step(k) #选择奖励的期望上置信界最大的一个
        self.estimate[k]+=1./(self.counts[k]+1)*(r-self.estimate[k])
        return k

    def run(self,run_nums):
        for _ in range(run_nums):
            k=self.run_one_step()
            self.counts[k]+=1
            self.update_regret(k)
            self.action.append(k)

np.random.seed(1)
coef = 1  # 控制不确定性比重的系数
UCB_solver = UCB(B, coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super().__init__(bandit)
        self._a=np.ones(self.bandit.K) #_a表示获得奖励的次数列表，初始化全为1，
        self._b=np.ones(self.bandit.K) #_b表示没获得奖励的次数列表，初始化全为1，

    def run_one_step(self):
        sample=np.random.beta(self._a,self._b) #为列表里面的每一项作为beta分布的一个参数，生成同长度的一个列表
        k=np.argmax(sample)
        r=self.bandit.step(k) #如果奖励则step返回1，否则返回0
        self._a[k]+=r #抽到奖励则a[k]+1，否则b[k]+1
        self._b[k]+=(1.-r)
        return k #最后要返回k，否则会报错数组形状不匹配（未知原因）
    
np.random.seed(1)
thompson_sampling_solver=ThompsonSampling(B)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：',thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver],["ThompsonSampling"])