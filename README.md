# Understanding Machine Learning Methods

Welcome to my digital playground where I explore the world of machine learning. Join me as I delve into my favorite areas like Generative Models and Reinforcement Learning. I will share some of my notes and resources that I found helpful. I hope you find them useful as well.

## Understanding Diffusion and Score-Based Generative Models

Ever wondered how Diffusion and Score-Based Generative Models work? Check out my beginner-friendly guide on [Score-Based and Denoising Diffusion Models](https://fema42.github.io/intro_to_diffusion/intro.html), where I break it down using Jupyter books. This introduction is inspired by amazing resources such as:

- **Diffusion-Denoising Models:** For a very good introduction into diffusion models you can check out the blog post from Lilian Weng: [Diffusion-Denoising Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/). Her blog is also a treasure trove on understandable guides of machine learning. Highly recommend visiting Lilian Weng's Blog: [Lilian Weng's Blog](https://lilianweng.github.io/lil-log/)

- **Score-Based Generative Models** Yang Song's introduction is an excellent starting point: [Score-Based Generative Models](https://yang-song.net/blog/2021/score/). His talk on [Youtube](https://www.youtube.com/watch?v=wMmqCMwuM2Q) on Score-Based Generative Models is also very insightful.

- **Comprehensive Denoising Diffusion-based Generative Modeling Tutorial:** At CVPR 2022 there was a great Tutorial on [Denoising Diffusion-based Generative Modeling - Foundations and Applications](https://cvpr2022-tutorial-diffusion-models.github.io/).

## Understanding Reinforcement Learning

Reinforcement Learning (RL) considers sequential decision making problems in a Markov decision processes (MDP). An agent interacts with an environment by choosing an action $a_t$ in each state $s_t$. Depending on the action $a_t$ in the state $s_t$ the environment emits a next state $s_{t+1}$ based on the transition dynamics $p(s_{t+1}|s_t,a_t)$ and a reward $r(s_t,a_t)$. The goal of the agent is to maximize the expected return $R$. The return is defined as the accumulated reward $R=\sum_{t=0}^T r(s_t,a_t)$. Note, that MDPs can be finite or infinete. For infinite MDPs we can use a discount factor $\gamma$ to ensure that the return is finite $R=\sum_{t=0}^{\infty} \gamma r(s_t,a_t)$. Several Methods have been developed to solve RL problems. To better understand these methods I list some resources that I found helpful.

[Reinforcement Learning - An Introduction](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) by Richard S. Sutton and Andrew G. Barto is a classic and a great in depth introduction into RL and MDPs. For a more practical introduction into RL I can recomment the following websites that also introduce mordern and commonly used RL algorithms:
- [Welcome to Spinning Up in Deep RL! — Spinning Up documentation](https://spinningup.openai.com/en/latest/)
- [Welcome to the 🤗 Deep Reinforcement Learning Course - Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course/en/unit0/introduction)

Explore practical implementations of RL algorithms with the following repositories:
- [CleanRL (Clean Implementation of RL Algorithms)](https://github.com/vwxyzjn/cleanrl) is doing a great job in providing a clean implementation of RL algorithms, where each algorithm if implemented in a single file.
- [Stable-Baselines](https://stable-baselines3.readthedocs.io/en/master/) implements many RL algorithms in PyTorch that are proven to work well, and thereby builds a great framework for stable RL implementations.

**Blog Post on debugging RL:**
Debugging RL can be hard (and sometimes even frustrating). I found the following blog post very helpful (and actually also very entertaining): [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html). Note that the article is from 2018 but I find many points are still valid today and the article is a great read. Another great article on debugging RL is: [Debugging Reinforcement Learning Systems](https://andyljones.com/posts/rl-debugging.html)

**Lecture on RL:**
If you want an even more in-depth introduction into RL, I can recommend the following lecture on RL: [CS 285 at UC Berkeley](https://rail.eecs.berkeley.edu/deeprlcourse/)

### Model-Based Reinforcement Learning

Model-based reinforcement learning (MBRL) also considers sequential decision making problems in a Markov decision processes. With States $s \in S$, actions $a \in A$, transition dynamics $p(s_{t+1}|s_t,a_t)$, reward $r(s_t,a_t)$ and if applicable horizon $T$. Contrary to model free RL we consider having or learning a *dynamics model*, to reason about the world. The dynamics model usually models the environment transition dynamics $s_{t+1}=f_{\psi}(s_t,a_t)$. Hence, the agent can use this dynamics model to decide how to act by predicting the future. Note that the true dynamics are often considered stochastic (that is why we used $p(\cdot)$ above). We can therefore also learn a stochastic model of our Environment.

If the model is learnable, the agent can collect more data to improve the model. How the agent uses the model to decide which action to take is based on its approach to planning. The agent can also learn a policy to predict actions (similar to model free RL approaches) to improve its planning based on experiences. Similarly other estimates like inverse dynamics models (mapping from states to actions) or reward models (predicting rewards) can be useful in this framework. One example to planning is [model-predictive control (MPC)](https://en.wikipedia.org/wiki/Model_predictive_control). Where the method optimizes the expected reward by searching the best actions. The actions are sampled for example using a uniformly distributed set of actions.

You can read more on model-based RL in a blog on [Debugging Deep Model-based Reinforcement Learning Systems](https://www.natolambert.com/writing/debugging-mbrl) and a recent survey on [Model-based Reinforcement Learning: A Survey](https://arxiv.org/abs/2006.16712).

### Imitation Learning

Imitation learning (IL) describes methods that learn optimal behavior that is represented by a collection of expert demonstrations. In IL, the agent also interacts with an environment in a sequential decision making process and therefore methods from RL can help to effectively solve IL problems. However, different to the RL-setting, does the agent not receive a reward from the environment. Instead, IL assumes that the experience comes from an expert policy (which behaves perfectly considering the task).  
Therefore, IL can alleviate the problem of designing effective reward functions. This is particularly useful for tasks where demonstrations are more accessible than designing a reward function. One example is to [train traffic agents in a simulation to mimic real-world road users](https://ieeexplore.ieee.org/document/9669229). In this case, it is easier to collect demonstrations of real-world road users than to design a reward function that captures all aspects of the task. A great overview of Imitation Learning is given in ["An Algorithmic Perspective on Imitation Learning"](https://arxiv.org/abs/1811.06711). A recent Imitation Learning method is [IQ-Learn](http://ai.stanford.edu/blog/learning-to-imitate/#inverse-q-learning-iq-learn) which is based on soft Q-Learning and learns a Q-function using the demonstration data. The authors showed that there is a one-to-one mapping between the learned Q-function and the underlying reward function and they can sucessfully estimate an reward based on the learned Q-function. We developed a method which does not require actions to be available in the expert data and can be used with state-only demonstrations. The method is called [Imitation Learning by State-Only Distribution Matching](https://arxiv.org/abs/2202.04332) and uses an expert and environment models to capture the reward function. Another great resource is the repository of [OPOLO: Off-policy Learning from Observations](https://github.com/illidanlab/opolo-code) which implements many IL methods.

### Offline Reinforcement Learning

Similar to IL is Offline Reinforcement Learning (Offline RL) a type of RL in which the agent learns from a dataset of previously collected experiences. However in Offline RL the agent learns without interacting with the environment during training. This makes it different from online RL, in which the agent learns while interacting with the environment. In IL both approaches are possible, while methods that rely on online interaction generally perform better. Both offline RL and IL rely on a dataset of experiences to learn from. They are, however, not the same. While Imitation Learning $(s_t, a_t, s_{t+1},)$ assumes that the experience comes from an expert policy (which behaves perfectly considering the task), Offline RL assumes that the experiences come from any policy and often the dataset contains also the Reward of the Environment $(s_t, a_t, s_{t+1}, r_t)$. So while these methods seem very similar the application of either method should be considered based on the task and availability of the data. 

For example imitation learning may be better suited for learning traffic agents to behave like real-world road users. The reason is that every recorded road user is by definition an "expert on human driving". However, if the task is to learn to drive a car as safe as possible, then offline RL is probably better suited. The reason is that the dataset of human driving behavior is not necessarily optimal for the task of driving as safe as possible. There is a good article discussing this topic: [Should I Use Offline RL or Imitation Learning?](https://bair.berkeley.edu/blog/2022/04/25/rl-or-bc/). Using Offline RL can help to learn a policy to perform a task before it is deployed in the real-world, where it might be dangerous to learn a policy online. A good overview on Offline RL is given in ["Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems"](https://arxiv.org/abs/2005.01643). Sergey Levin and Aviral Kumar gave a great tutorial at [NeurIPS 2020](https://sites.google.com/view/offlinerltutorial-neurips2020/home) on Offline RL.

## Understanding Numerical Methods

I held the exercises for the course "Numerical Methods for Engineers" at the University of Augsburg, Germany. I've made a [jupyter book](https://fema42.github.io/numerical_methods/intro.html) out of the exercises and the exercises itself are availale at: [Numerical Methods for Engineers](https://github.com/FeMa42/auxme_numerik.git) as Jupyter notebooks (partly written in German) with julia code. Many of my exerrcise are based on the great book [Fundamentals of Numerical Computation](https://tobydriscoll.net/fnc-julia/home.html) by Toby A. Driscoll and Richard J. Braun.

## Contact

I am a PhD student at the University of Augsburg, Germany. I am part of the [Chair of Mechatronics](https://www.uni-augsburg.de/de/fakultaet/fai/informatik/prof/imech/team/damian-boborzi/) at the Faculty of Applied Computer Science. My research interests lie in the field of machine learning, specifically in the areas of generative models and reinforcement learning.

If you have any questions or comments, feel free to reach out to me via mail: damian.boborzi@uni-a.de.
