import os
from re import X
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

if __name__ == '__main__':
    from src.agents import UCB1Agent, Exp3Agent, MiniBatchExp3Agent, AutoregressiveRidgeAgent, AutoregressiveClairvoyant, AR2Agent
    from src.environment import AutoregressiveEnvironment
    from src.core import Core
    import matplotlib.pyplot as plt
    import numpy as np
    import tikzplotlib as tikz
    import warnings
    import json
    import sys

    warnings.filterwarnings('ignore')
    final_logs = {}
    simulation_id = sys.argv[1]
    out_folder = f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/output/simulation__baselines2_{simulation_id}/'
    try:
        os.mkdir(out_folder)
        os.mkdir(out_folder + 'tex/')
        os.mkdir(out_folder + 'png/')
    except:
        pass
        #raise NameError(f'Folder {out_folder} already exists.')
    
    graphName = ""

    for testcase_id in sys.argv[2:]:
        # testcase_id = int(testcase_id)
        print(f'################## Testcase {testcase_id} ###################')

        f = open(f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/input/{simulation_id}_{testcase_id}.json')
        param_dict = json.load(f)

        input_testcase = str(testcase_id)

        testcase = input_testcase.split('=', 1)[0]
        
        print(f'Parameters: {param_dict}')

        param_dict['gamma']=np.array(param_dict['gamma'])

        T = param_dict['T']
        k = param_dict['gamma'].shape[1]-1
        n_arms = param_dict['gamma'].shape[0]
        
        logs = {}
        a_hists = {}

        width=4
        height=4

        if testcase == "Gamma6":
            top_bound=10000
        elif testcase == "Gamma8":
            top_bound=5e6
        elif testcase == "Gamma11":
            top_bound=1e7
        else:
            top_bound=100000

        #calculating Gamma

        gamma_array=[]

        for key in param_dict['gamma']:
            gamma_array = np.append(gamma_array,np.cumsum(key)[-1] - key[0])

        #print(gamma_array)

        Gamma=np.round(gamma_array.max(),3)

        #print(Gamma)

        # Clairvoyant
        print('Training Clairvoyant algorithm')
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs, a_hists['Clairvoyant'] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs = clairvoyant_logs[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs.max()

        # do the following by Example: replace logs['name'], a_hists['name'] with logs['name_'testcase_id], a_hists['name_'testcase_id]

        # UCB1
        print('Training UCB1 Algorithm')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = UCB1Agent(n_arms, sigma=param_dict['noise_std'])
        core = Core(env, agent)
        logs['UCB1_'+testcase_id], a_hists['UCB1_'+testcase_id] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs['UCB1_'+testcase_id] = logs['UCB1_'+testcase_id][:, len(param_dict['X0']):]

        # EXP3 
        print('Training EXP3 Algorithm')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        lr = min(1, np.sqrt(n_arms*np.log(n_arms)/((np.exp(1)-1)*T)))
        agent = Exp3Agent(n_arms, gamma=lr, max_reward=max_reward, random_state=param_dict['seed'])
        core = Core(env, agent)
        logs['EXP3_'+testcase_id], a_hists['EXP3_'+testcase_id] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs['EXP3_'+testcase_id] = logs['EXP3_'+testcase_id][:, len(param_dict['X0']):]

        # MiniBatchEXP3 
        print('Training MiniBatchEXP3 Algorithm')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        C = 7*n_arms*np.log(n_arms)
        batch_size = int(C**(-1/3)*T**(1/3))
        lr = min(1, np.sqrt(n_arms*np.log(n_arms)/((np.exp(1)-1)*(T/batch_size))))
        lr = 0.14
        agent = MiniBatchExp3Agent(n_arms, gamma=lr, max_reward=max_reward, batch_size=batch_size, random_state=param_dict['seed'])
        core = Core(env, agent)
        logs['MiniBatchEXP3_'+testcase_id], a_hists['MiniBatchEXP3_'+testcase_id] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs['MiniBatchEXP3_'+testcase_id] = logs['MiniBatchEXP3_'+testcase_id][:, len(param_dict['X0']):]

        # AR2
        sigma = param_dict['noise_std']
        if param_dict['gamma'].shape[1]>1:
            alpha = max(np.sum(param_dict['gamma'][:,1:],axis=1))
            epoch_size = int(n_arms/(alpha*sigma)**3+1)
            c0 = np.sqrt(4*np.log(1/(alpha*sigma))+4*np.log(epoch_size)+2*np.log(4*n_arms))
        else:
            epoch_size = T
            alpha = 0
            c0 = 0
        print('Training AR2 Algorithm')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AR2Agent(n_arms, alpha=alpha, epoch_size=epoch_size, c0=c0, sigma=sigma)
        core = Core(env, agent)
        logs['AR2_'+testcase_id], a_hists['AR2_'+testcase_id] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs['AR2_'+testcase_id] = logs['AR2_'+testcase_id][:, len(param_dict['X0']):]
        
        # ARB
        print('Training ARB Algorithm')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs['ARB_'+testcase_id], a_hists['ARB_'+testcase_id] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs['ARB_'+testcase_id] = logs['ARB_'+testcase_id][:, len(param_dict['X0']):]

        # Regrets computing
        print('Computing regrets...')
        regret = {label : np.inf * np.ones((param_dict['n_epochs'], T)) for label in logs.keys()}
        for i in range(param_dict['n_epochs']):
            for label in regret.keys():
                regret[label][i, :] = clairvoyant_logs[i, :] - logs[label][i, :]

        # graphName=graphName+"_"+testcase_id

        # # inst reward, inst regret and cumulative regret plot

        # x = np.arange(len(param_dict['X0']),T+1, step=250)
        # f,ax = plt.subplots(3, figsize=(20,30))
        # sqrtn = np.sqrt(param_dict['n_epochs'])

        # ax[0].plot(x, np.mean(clairvoyant_logs.T, axis=1)[x], label=clrv, color='C0')
        # ax[0].fill_between(x, np.mean(clairvoyant_logs.T, axis=1)[x]-np.std(clairvoyant_logs.T, axis=1)[x]/sqrtn,
        #                 np.mean(clairvoyant_logs.T, axis=1)[x]+np.std(clairvoyant_logs.T, axis=1)[x]/sqrtn, alpha=0.3, color='C0')
        # for i,label in enumerate(regret.keys()):
        #     ax[0].plot(x, np.mean(logs[label].T,axis=1)[x], label=label, color=f'C{i+1}')
        #     ax[0].fill_between(x, np.mean(logs[label].T, axis=1)[x]-np.std(logs[label].T, axis=1)[x]/sqrtn, 
        #                     np.mean(logs[label].T, axis=1)[x]+np.std(logs[label].T, axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')   
        #     ax[1].plot(x, np.mean(regret[label].T, axis=1)[x], label=label, color=f'C{i+1}')
        #     ax[1].fill_between(x, np.mean(regret[label].T, axis=1)[x]-np.std(regret[label].T, axis=1)[x]/sqrtn,
        #                 np.mean(regret[label].T, axis=1)[x]+np.std(regret[label].T, axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')
        #     ax[2].plot(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x], label=label, color=f'C{i+1}')
        #     ax[2].fill_between(x, np.mean(np.cumsum(regret[label].T, axis=0),axis=1)[x]-np.std(np.cumsum(regret[label].T, axis=0),axis=1)[x]/sqrtn,
        #                 np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')
            
        # ax[0].set_xlim(left=0)
        # ax[0].set_title('Instantaneous Rewards')
        # ax[0].legend()

        # ax[1].set_xlim(left=0)
        # ax[1].set_title('Instantaneous Regret')
        # ax[1].legend()

        # ax[2].set_xlim(left=0)
        # ax[2].set_title('Cumulative Regret')
        # ax[2].legend()

        # tikz.save(out_folder + f"tex/{simulation_id}{graphName}_baselines_all.tex")
        # # tikz.save(out_folder + f"tex/text.tex")
        # plt.savefig(out_folder + f"png/{simulation_id}{graphName}_baselines_all.png")
        # # plt.savefig(out_folder + f"png/pic.png")

        #  cumulative regret plot

        x = np.arange(len(param_dict['X0']),T+50, step=50)
        x[-1] = min(x[-1],len(np.mean(np.cumsum(regret['ARB_'+testcase_id].T, axis=0), axis=1))-1)
        f,ax = plt.subplots(1, figsize=(width,height))
        sqrtn = np.sqrt(param_dict['n_epochs'])

        for i,label in enumerate(regret.keys()):
            ax.plot(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x], label=label.replace("_"+testcase_id,""), color=f'C{2*i+1}')
            ax.fill_between(x, np.mean(np.cumsum(regret[label].T, axis=0),axis=1)[x]-np.std(np.cumsum(regret[label].T, axis=0),axis=1)[x]/sqrtn,
                        np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3, color=f'C{2*i+1}')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=top_bound)
        ax.set_title('Cumulative Regrets: '+'\u0393'+"="+str(Gamma))
        ax.legend()    

        tikz.save(out_folder + f"tex/{simulation_id}{testcase_id}_baselines_regret_2.tex")
        plt.savefig(out_folder + f"png/{simulation_id}{testcase_id}_baselines_regret_2.png")

        # logging 

        final_logs[f'{simulation_id}_{testcase_id}'] = {label : np.mean(np.sum(regret[label].T, axis=0)) for label in regret.keys()}

        # action history plots

        f,ax = plt.subplots(3,2, figsize=(20,30))

        for ax_, label in zip(f.axes, a_hists.keys()):
            bins = np.arange(n_arms+1) - 0.5
            ax_.hist(a_hists[label].flatten(), bins=bins)
            ax_.set_xticks(range(n_arms))
            ax_.set_xlim(-1, n_arms)
            ax_.set_title(label)
        
        tikz.save(out_folder + f"tex/{simulation_id}_{testcase_id}_a_hist_baselines.tex")
        plt.savefig(out_folder + f"png/{simulation_id}_{testcase_id}_a_hist_baselines.png")

        f,ax = plt.subplots(3,2, figsize=(20,30))

        for ax_, label in zip(f.axes, a_hists.keys()):
            bins = np.arange(n_arms+1) - 0.5
            ax_.plot(a_hists[label][-1,:])
            ax_.set_title(label)
        
        tikz.save(out_folder + f"tex/{simulation_id}_{testcase_id}_a_hist_temp_baselines.tex")
        plt.savefig(out_folder + f"png/{simulation_id}_{testcase_id}_a_hist_temp_baselines.png")

    out_file = open(out_folder+f"logs_exp.json", "w")    
    json.dump(final_logs, out_file, indent = 4)
    out_file.close() 
            
