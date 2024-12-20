import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

from memory import *
from actor_critic_networks import *
from noise_process import *


Sequence = namedtuple("Sequence", ["s", "a", "r", "s_", "done"])

def update_grads(buffer, actor, critic, actor_tar, critic_tar, opt_mu, opt_q):

    tau=0.001
    gm=0.99
    batch_size=64
    batch=buffer.sample(batch_size)

    #update critic grads
    with torch.no_grad():
        #Q(s_,a_)
        Q_next=critic_tar(batch.s_, actor_tar(batch.s_))
        #~ take negation, it means if done Q_tar=r otherwise Q_tar=r+gm*Q_next
        Q_tar=batch.r+gm*torch.mul(Q_next, (~batch.done).float()).detach()

    #update critic grads
    #Q(s,a)
    Q_cur=critic(batch.s, batch.a)
    #mse(Q(s,a),Q(s_,a_))
    q_loss=F.mse_loss(Q_cur, Q_tar)

    opt_q.zero_grad()
    q_loss.backward()
    opt_q.step()

    #update actor grads
    mu_loss=-critic(batch.s, actor(batch.s)).mean()
    opt_mu.zero_grad()
    mu_loss.backward()
    opt_mu.step()

    #soft update
    for p, p_tar in zip(critic.parameters(), critic_tar.parameters()):
        p_tar.data.copy_(tau * p.data + (1 - tau) * p_tar.data)

    for p, p_tar in zip(actor.parameters(), actor_tar.parameters()):
        p_tar.data.copy_(tau * p.data + (1 - tau) * p_tar.data)

    return q_loss.item(), mu_loss.item()

def train():

    #hyperparameters
    lr_mu=0.0001
    lr_q=0.001
    gm=0.99
    tau=0.001
    n_eps=2500
    min_memo_size=2000
    buffer_size=1000000
    #ou noise
    theta,sigma,decay,sigma_min=0.15,0.2,0.0,0.15

    env=gym.make("LunarLanderContinuous-v2")

    n_s=env.observation_space.shape[0]
    n_a=env.action_space.shape[0]

    #initalization
    actor=Actor(n_s, env.action_space, 400, 400) #(layer1,layer2) sizes (400,400)
    critic=Critic(n_s, n_a, 400, 400)
    actor_tar=Actor(n_s, env.action_space, 400, 400) #(layer1,layer2) sizes (400,400)
    critic_tar=Critic(n_s, n_a, 400, 400)
    actor_tar.load_state_dict(actor.state_dict())
    critic_tar.load_state_dict(critic.state_dict())

    opt_mu=optim.Adam(actor.parameters(), lr_mu)
    opt_q=optim.Adam(critic.parameters(), lr_q)

    buffer=Memory(buffer_size)
    noise=NoiseProcess(env.action_space,theta,sigma,decay,sigma_min)

    r_all,stp_all=[],[]

    for ep in range(n_eps):
        s, _ = env.reset()
        done = False
        r_sum, stps = 0, 0

        while not done and stps < env._max_episode_steps:
            stps += 1
            noiz = noise.sample()
            a = actor.take_action(s,noiz)
            s_,r,done,trun,_ = env.step(a)
            buffer.push(Sequence(s,a,r,s_,done))
            s = s_
            r_sum += r

            if buffer.max_entry > min_memo_size:
                mu_loss, q_loss = update_grads(buffer, actor, critic, actor_tar, critic_tar, opt_mu, opt_q)

        noise.decay()
        r_all.append(r_sum)
        stp_all.append(stps)
        print(f"Ep:{ep} Stps:{stps} R:{r_sum}")

    np.save('r_all.npy',r_all)
    np.save('stp_all.npy',stp_all)

if __name__ == "__main__":

    train()
