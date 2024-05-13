import os
from re import X
os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'

if __name__ == '__main__':
    from src.agents import AutoregressiveRidgeAgent, AutoregressiveClairvoyant
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
    out_folder = f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/output/simulation_{simulation_id}_3/'
    try:
        os.mkdir(out_folder)
        os.mkdir(out_folder + 'tex/')
        os.mkdir(out_folder + 'png/')
    except:
        pass
        #raise NameError(f'Folder {out_folder} already exists.')

    # logs = {}
    # a_hists = {}
    # m_logs = []

    m_values = [0, 0.25, 0.5, 1, 10, 100, 500, 1000]

    for testcase_id in sys.argv[2:]:
        # testcase_id = int(testcase_id)
        print(f'################## Experimental Testcase {testcase_id} ###################')

        f = open(f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/input/{simulation_id}_{testcase_id}.json')
        param_dict = json.load(f)
        
        print(f'Initial Parameters: {param_dict}')

        param_dict['gamma']=np.array(param_dict['gamma'])

        logs = {}
        a_hists = {}

        T = param_dict['T']
        k = param_dict['gamma'].shape[1]-1
        n_arms = param_dict['gamma'].shape[0]

        #setting graph parameters

        top_bound = 0

        width=4
        height=4

        # if testcase_id == "Gamma8=095":
        #     top_bound=1e6
        # elif testcase_id == "Gamma8=098":
        #     top_bound=1e6
        # elif testcase_id == "Gamma8=0999":
        #     top_bound=1e7
        # elif testcase_id == "Gamma8=1":
        #     top_bound=1e7
        # else:
        #     top_bound=100000

        print("Bound: "+str(top_bound))

        #Computing Gamma

        gamma_array=[]

        for key in param_dict['gamma']:
            gamma_array = np.append(gamma_array,np.cumsum(key)[-1] - key[0])

        Gamma=np.round(gamma_array.max(),3)

        print('Gamma='+str(Gamma))

        #m=0

        param_dict['m']=0

        for key in param_dict['gamma']:
            key[0]=0

        # Clairvoyant
        print('Training Clairvoyant algorithm with m=0')
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_0, a_hists['Clairvoyant'+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_0 = clairvoyant_logs_0[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_0.max()

         # ARB
        print('Training ARB Algorithm with m=0')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['m'])], a_hists[testcase_id+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['m'])] = logs[testcase_id+str(param_dict['m'])][:, len(param_dict['X0']):]

        #m=0.25

        param_dict['m']=0.25

        for key in param_dict['gamma']:
            key[0]=0.25

        # Clairvoyant
        print('Training Clairvoyant algorithm with m=0.25')
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_025, a_hists['Clairvoyant'+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_025 = clairvoyant_logs_025[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_025.max()

         # ARB
        print('Training ARB Algorithm with m=0.25')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['m'])], a_hists[testcase_id+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['m'])] = logs[testcase_id+str(param_dict['m'])][:, len(param_dict['X0']):]

        #m=0.5

        param_dict['m']=0.5

        for key in param_dict['gamma']:
            key[0]=0.5

        # Clairvoyant
        print('Training Clairvoyant algorithm with m=0.5')
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_05, a_hists['Clairvoyant'+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_05 = clairvoyant_logs_05[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_05.max()

         # ARB
        print('Training ARB Algorithm with m=0.5')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['m'])], a_hists[testcase_id+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['m'])] = logs[testcase_id+str(param_dict['m'])][:, len(param_dict['X0']):]

        #m=1

        param_dict['m']=1

        for key in param_dict['gamma']:
            key[0]=1

        # Clairvoyant
        print('Training Clairvoyant algorithm with m=1')
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_1, a_hists['Clairvoyant'+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_1 = clairvoyant_logs_1[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_1.max()

         # ARB
        print('Training ARB Algorithm with m=1')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['m'])], a_hists[testcase_id+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['m'])] = logs[testcase_id+str(param_dict['m'])][:, len(param_dict['X0']):]

        #m=10

        param_dict['m']=10

        for key in param_dict['gamma']:
            key[0]=10

        # Clairvoyant
        print('Training Clairvoyant algorithm with m=10')
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_10, a_hists['Clairvoyant'+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_10 = clairvoyant_logs_10[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_10.max()

         # ARB
        print('Training ARB Algorithm with m=10')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['m'])], a_hists[testcase_id+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['m'])] = logs[testcase_id+str(param_dict['m'])][:, len(param_dict['X0']):]

        #m=100

        param_dict['m']=100

        for key in param_dict['gamma']:
            key[0]=100

        # Clairvoyant
        print('Training Clairvoyant algorithm with m=100')
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_100, a_hists['Clairvoyant'+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_100 = clairvoyant_logs_100[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_100.max()

         # ARB
        print('Training ARB Algorithm with m=100')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['m'])], a_hists[testcase_id+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['m'])] = logs[testcase_id+str(param_dict['m'])][:, len(param_dict['X0']):]

        #m=500

        param_dict['m']=500

        for key in param_dict['gamma']:
            key[0]=500

        # Clairvoyant
        print('Training Clairvoyant algorithm with m=500')
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_500, a_hists['Clairvoyant'+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_500 = clairvoyant_logs_500[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_500.max()

         # ARB
        print('Training ARB Algorithm with m=500')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['m'])], a_hists[testcase_id+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['m'])] = logs[testcase_id+str(param_dict['m'])][:, len(param_dict['X0']):]

        #m=1000

        param_dict['m']=1000

        for key in param_dict['gamma']:
            key[0]=1000

        # Clairvoyant
        print('Training Clairvoyant algorithm with m=1000')
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_1000, a_hists['Clairvoyant'+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_1000 = clairvoyant_logs_1000[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_1000.max()

         # ARB
        print('Training ARB Algorithm with m=1000')
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['m'])], a_hists[testcase_id+str(param_dict['m'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['m'])] = logs[testcase_id+str(param_dict['m'])][:, len(param_dict['X0']):]

        # Regrets computing
        print('Computing regrets...')
        regret = {label : np.inf * np.ones((param_dict['n_epochs'], T)) for label in logs.keys()}
        for i in range(param_dict['n_epochs']):
            regret[testcase_id+str(0)][i, :] = clairvoyant_logs_0[i, :] - logs[testcase_id+str(0)][i, :]
            regret[testcase_id+str(0.25)][i, :] = clairvoyant_logs_025[i, :] - logs[testcase_id+str(0.25)][i, :]
            regret[testcase_id+str(0.5)][i, :] = clairvoyant_logs_05[i, :] - logs[testcase_id+str(0.5)][i, :]
            regret[testcase_id+str(1)][i, :] = clairvoyant_logs_1[i, :] - logs[testcase_id+str(1)][i, :]
            regret[testcase_id+str(10)][i, :] = clairvoyant_logs_10[i, :] - logs[testcase_id+str(10)][i, :]
            regret[testcase_id+str(100)][i, :] = clairvoyant_logs_100[i, :] - logs[testcase_id+str(100)][i, :]
            regret[testcase_id+str(500)][i, :] = clairvoyant_logs_500[i, :] - logs[testcase_id+str(500)][i, :]
            regret[testcase_id+str(1000)][i, :] = clairvoyant_logs_1000[i, :] - logs[testcase_id+str(1000)][i, :]
    
        print('Creating Graphs...')

        # graphName=graphName+"_"+testcase_id

        # inst reward, inst regret and cumulative regret plot - replace f and ax with f_cm and ax_cm

        # x = np.arange(len(param_dict['X0']),T+1, step=250)
        # f_cm,ax_cm = plt.subplots(3, figsize=(20,30))
        # sqrtn = np.sqrt(param_dict['n_epochs'])

        # ax_cm[0].plot(x, np.mean(clairvoyant_logs.T, axis=1)[x], label=clrv, color='C0')
        # ax_cm[0].fill_between(x, np.mean(clairvoyant_logs.T, axis=1)[x]-np.std(clairvoyant_logs.T, axis=1)[x]/sqrtn,
        #                 np.mean(clairvoyant_logs.T, axis=1)[x]+np.std(clairvoyant_logs.T, axis=1)[x]/sqrtn, alpha=0.3, color='C0')
        # for i,label in enumerate(regret.keys()):
        #     # for testcase_id in sys.argv[2:]:
        #     ax_cm[0].plot(x, np.mean(logs[label].T,axis=1)[x], label=label, color=f'C{i+1}')
        #     ax_cm[0].fill_between(x, np.mean(logs[label].T, axis=1)[x]-np.std(logs[label].T, axis=1)[x]/sqrtn, 
        #                     np.mean(logs[label].T, axis=1)[x]+np.std(logs[label].T, axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')   
        #     ax_cm[1].plot(x, np.mean(regret[label].T, axis=1)[x], label=label, color=f'C{i+1}')
        #     ax_cm[1].fill_between(x, np.mean(regret[label].T, axis=1)[x]-np.std(regret[label].T, axis=1)[x]/sqrtn,
        #                 np.mean(regret[label].T, axis=1)[x]+np.std(regret[label].T, axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')
        #     ax_cm[2].plot(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x], label=label, color=f'C{i+1}')
        #     ax_cm[2].fill_between(x, np.mean(np.cumsum(regret[label].T, axis=0),axis=1)[x]-np.std(np.cumsum(regret[label].T, axis=0),axis=1)[x]/sqrtn,
        #                 np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')

        # ax_cm[0].set_xlim(left=0)
        # ax_cm[0].set_title('ARB-Instantaneous Rewards')
        # ax_cm[0].legend()

        # ax_cm[1].set_xlim(left=0)
        # ax_cm[1].set_title('ARB-Instantaneous Regret')
        # ax_cm[1].legend()

        # ax_cm[2].set_xlim(left=0)
        # ax_cm[2].set_title('ARB-Cumulative Regret')
        # ax_cm[2].legend()

        # tikz.save(out_folder + f"tex/{simulation_id}{graphName}_all.tex")
        # plt.savefig(out_folder + f"png/{simulation_id}{graphName}_all.png")

        #  cumulative regret plot - replace f and ax with f_reg and ax_reg
        
        f_reg,ax_reg = plt.subplots(1, figsize=(width,height))
        x = np.arange(len(param_dict['X0']),T+50, step=50)
        x[-1] = min(x[-1],len(np.mean(np.cumsum(regret[testcase_id+str(0)].T, axis=0), axis=1))-1)
        sqrtn = np.sqrt(param_dict['n_epochs'])
        for i,label in enumerate(regret.keys()): 
            ax_reg.plot(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x], label="m="+str(m_values[i]), color=f'C{i+2}')
            ax_reg.fill_between(x, np.mean(np.cumsum(regret[label].T, axis=0),axis=1)[x]-np.std(np.cumsum(regret[label].T, axis=0),axis=1)[x]/sqrtn,
                    np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+2}')
            ax_reg.set_xlim(left=0)
            ax_reg.set_ylim(bottom=0, top=2e6)
            ax_reg.set_title('ARB-Cumulative Regret')
            ax_reg.legend()    

        tikz.save(out_folder + f"tex/{simulation_id}_{testcase_id}_regret_m_{Gamma}_1.tex")
        plt.savefig(out_folder + f"png/{simulation_id}_{testcase_id}_regret_m_{Gamma}-1.png")

        # logging 

        final_logs[f'testcase_{testcase_id}'] = {label : np.mean(np.sum(regret[label].T, axis=0)) for label in regret.keys()}

        # action history plots

        f,ax = plt.subplots(3,2, figsize=(20,30))

        for ax_, label in zip(f.axes, a_hists.keys()):
            bins = np.arange(n_arms+1) - 0.5
            ax_.hist(a_hists[label].flatten(), bins=bins)
            ax_.set_xticks(range(n_arms))
            ax_.set_xlim(-1, n_arms)
            ax_.set_title(label)
        
        tikz.save(out_folder + f"tex/{simulation_id}_{testcase_id}_a_hist.tex")
        plt.savefig(out_folder + f"png/{simulation_id}_{testcase_id}_a_hist.png")

        f,ax = plt.subplots(3,2, figsize=(20,30))

        for ax_, label in zip(f.axes, a_hists.keys()):
            bins = np.arange(n_arms+1) - 0.5
            ax_.plot(a_hists[label][-1,:])
            ax_.set_title(label)
        
        tikz.save(out_folder + f"tex/{simulation_id}_{testcase_id}_a_hist_temp.tex")
        plt.savefig(out_folder + f"png/{simulation_id}_{testcase_id}_a_hist_temp.png")
        # plt.show()

    out_file = open(out_folder+f"logs_exp.json", "w")    
    json.dump(final_logs, out_file, indent = 4)
    out_file.close()
