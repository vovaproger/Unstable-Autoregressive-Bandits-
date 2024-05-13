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

    lambda_logs = [0.0001,0.01,0.05,0.2,0.6,1,1.2,1.6,3,5]

    for testcase_id in sys.argv[2:]:
        # testcase_id = int(testcase_id)
        print(f'################## Experimental Testcase {testcase_id} ###################')

        f = open(f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/input/{simulation_id}_{testcase_id}.json')
        param_dict = json.load(f)
        
        print(f'Parameters: {param_dict}')

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

        testcase = testcase_id.split('_')

        # if testcase_id == "Gamma8=095" or testcase_id=="Gamma8=095_m=1":
        #     top_bound=80000
        # elif testcase_id == "Gamma8=098" or testcase_id=="Gamma8=098_m=1":
        #     top_bound=100000
        # elif testcase_id == "Gamma8=0999" or testcase_id == "Gamma8=0999_m=1":
        #     top_bound=2e6
        # elif testcase_id=="Gamma8=1" or testcase_id=="Gamma8=1_m=1":
        #     top_bound=5e6
        # else:
        #     top_bound=100000

        if len(testcase) == 2:
            top_bound = 750000
        else:
            top_bound = 1e6

        print("Bound: "+str(top_bound))

        #lambda=0.0001

        param_dict['lambda'] = 0.0001
        
        # Clairvoyant
        print('Training Clairvoyant algorithm with λ='+str(param_dict['lambda']))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_00001, a_hists['Clairvoyant'+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_00001 = clairvoyant_logs_00001[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_00001.max()

        # ARB
        print('Training ARB Algorithm with λ='+str(param_dict['lambda']))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['lambda'])], a_hists[testcase_id+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['lambda'])] = logs[testcase_id+str(param_dict['lambda'])][:, len(param_dict['X0']):]

        #lambda=0.01

        param_dict['lambda'] = 0.01
        
        # Clairvoyant
        print('Training Clairvoyant algorithm with λ='+str(param_dict['lambda']))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_001, a_hists['Clairvoyant'+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_001 = clairvoyant_logs_001[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_001.max()

        # ARB
        print('Training ARB Algorithm with λ='+str(param_dict['lambda']))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['lambda'])], a_hists[testcase_id+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['lambda'])] = logs[testcase_id+str(param_dict['lambda'])][:, len(param_dict['X0']):]

        #lambda=0.05

        param_dict['lambda'] = 0.05
        
        # Clairvoyant
        print('Training Clairvoyant algorithm with λ='+str(param_dict['lambda']))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_005, a_hists['Clairvoyant'+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_005 = clairvoyant_logs_005[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_005.max()

        # ARB
        print('Training ARB Algorithm with λ='+str(param_dict['lambda']))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['lambda'])], a_hists[testcase_id+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['lambda'])] = logs[testcase_id+str(param_dict['lambda'])][:, len(param_dict['X0']):]

        #lambda=0.2

        param_dict['lambda'] = 0.2
        
        # Clairvoyant
        print('Training Clairvoyant algorithm with λ='+str(param_dict['lambda']))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_02, a_hists['Clairvoyant'+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_02 = clairvoyant_logs_02[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_02.max()

        # ARB
        print('Training ARB Algorithm with λ='+str(param_dict['lambda']))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['lambda'])], a_hists[testcase_id+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['lambda'])] = logs[testcase_id+str(param_dict['lambda'])][:, len(param_dict['X0']):]
       #lambda=0.2

        param_dict['lambda'] = 0.6
        
        # Clairvoyant
        print('Training Clairvoyant algorithm with λ='+str(param_dict['lambda']))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_06, a_hists['Clairvoyant'+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_06 = clairvoyant_logs_06[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_06.max()

        # ARB
        print('Training ARB Algorithm with λ='+str(param_dict['lambda']))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['lambda'])], a_hists[testcase_id+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['lambda'])] = logs[testcase_id+str(param_dict['lambda'])][:, len(param_dict['X0']):]

        #lambda=1

        param_dict['lambda'] = 1
        
        # Clairvoyant
        print('Training Clairvoyant algorithm with λ='+str(param_dict['lambda']))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_1, a_hists['Clairvoyant'+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_1 = clairvoyant_logs_1[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_1.max()

        # ARB
        print('Training ARB Algorithm with λ='+str(param_dict['lambda']))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['lambda'])], a_hists[testcase_id+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['lambda'])] = logs[testcase_id+str(param_dict['lambda'])][:, len(param_dict['X0']):]

        #lambda=1.2

        param_dict['lambda'] = 1.2
        
        # Clairvoyant
        print('Training Clairvoyant algorithm with λ='+str(param_dict['lambda']))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_12, a_hists['Clairvoyant'+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_12 = clairvoyant_logs_12[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_12.max()

        # ARB
        print('Training ARB Algorithm with λ='+str(param_dict['lambda']))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['lambda'])], a_hists[testcase_id+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['lambda'])] = logs[testcase_id+str(param_dict['lambda'])][:, len(param_dict['X0']):]
        

        #lambda=1.6

        param_dict['lambda'] = 1.6
        
        # Clairvoyant
        print('Training Clairvoyant algorithm with λ='+str(param_dict['lambda']))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_16, a_hists['Clairvoyant'+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_16 = clairvoyant_logs_16[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_16.max()

        # ARB
        print('Training ARB Algorithm with λ='+str(param_dict['lambda']))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['lambda'])], a_hists[testcase_id+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['lambda'])] = logs[testcase_id+str(param_dict['lambda'])][:, len(param_dict['X0']):]
        
        #lambda=3

        param_dict['lambda'] = 3
        
        # Clairvoyant
        print('Training Clairvoyant algorithm with λ='+str(param_dict['lambda']))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_3, a_hists['Clairvoyant'+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_3 = clairvoyant_logs_3[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_3.max()

        # ARB
        print('Training ARB Algorithm with λ='+str(param_dict['lambda']))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['lambda'])], a_hists[testcase_id+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['lambda'])] = logs[testcase_id+str(param_dict['lambda'])][:, len(param_dict['X0']):]
        
        #lambda=5

        param_dict['lambda'] = 5
        
        # Clairvoyant
        print('Training Clairvoyant algorithm with λ='+str(param_dict['lambda']))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_5, a_hists['Clairvoyant'+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_5 = clairvoyant_logs_5[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_5.max()

        # ARB
        print('Training ARB Algorithm with λ='+str(param_dict['lambda']))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+str(param_dict['lambda'])], a_hists[testcase_id+str(param_dict['lambda'])] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+str(param_dict['lambda'])] = logs[testcase_id+str(param_dict['lambda'])][:, len(param_dict['X0']):]
        
        # Regrets computing
        print('Computing regrets...')
        regret = {label : np.inf * np.ones((param_dict['n_epochs'], T)) for label in logs.keys()}
        for i in range(param_dict['n_epochs']):
            regret[testcase_id+str(0.0001)][i, :] = clairvoyant_logs_00001[i, :] - logs[testcase_id+str(0.0001)][i, :]
            regret[testcase_id+str(0.01)][i, :] = clairvoyant_logs_001[i, :] - logs[testcase_id+str(0.01)][i, :]
            regret[testcase_id+str(0.05)][i, :] = clairvoyant_logs_005[i, :] - logs[testcase_id+str(0.05)][i, :]
            regret[testcase_id+str(0.2)][i, :] = clairvoyant_logs_02[i, :] - logs[testcase_id+str(0.2)][i, :]
            regret[testcase_id+str(0.6)][i, :] = clairvoyant_logs_06[i, :] - logs[testcase_id+str(0.6)][i, :]
            regret[testcase_id+str(1)][i, :] = clairvoyant_logs_1[i, :] - logs[testcase_id+str(1)][i, :]
            regret[testcase_id+str(1.2)][i, :] = clairvoyant_logs_12[i, :] - logs[testcase_id+str(1.2)][i, :]
            regret[testcase_id+str(1.6)][i, :] = clairvoyant_logs_16[i, :] - logs[testcase_id+str(1.6)][i, :]
            regret[testcase_id+str(3)][i, :] = clairvoyant_logs_3[i, :] - logs[testcase_id+str(3)][i, :]
            regret[testcase_id+str(5)][i, :] = clairvoyant_logs_5[i, :] - logs[testcase_id+str(5)][i, :]
    
        print('Creating Graphs...')

        #  cumulative regret plot - replace f and ax with f_reg and ax_reg
        
        f_reg,ax_reg = plt.subplots(1, figsize=(width, height))
        x = np.arange(len(param_dict['X0']),T+50, step=50)
        x[-1] = min(x[-1],len(np.mean(np.cumsum(regret[testcase_id+str(0.0001)].T, axis=0), axis=1))-1)
        sqrtn = np.sqrt(param_dict['n_epochs'])
        for i,label in enumerate(regret.keys()):
            
            ax_reg.plot(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x], label="λ="+str(lambda_logs[i]), color=f'C{i+1}')
            ax_reg.fill_between(x, np.mean(np.cumsum(regret[label].T, axis=0),axis=1)[x]-np.std(np.cumsum(regret[label].T, axis=0),axis=1)[x]/sqrtn,
                    np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3, color=f'C{i+1}')
            plt.subplots_adjust(left=0.15)
            ax_reg.set_xlim(left=0)
            ax_reg.set_ylim(bottom=0, top=top_bound) # 
            #ax_reg.adjust()
            ax_reg.set_title('ARB-Cumulative Regret')
            ax_reg.legend()    

        tikz.save(out_folder + f"tex/{simulation_id}_{testcase_id}_regret_3_lambda_m=1.tex")
        plt.savefig(out_folder + f"png/{simulation_id}_{testcase_id}_regret_3_lambda_m=1.png")

        # logging 

        # final_logs[f'testcase_{testcase_id}'] = {label : np.mean(np.sum(regret[label].T, axis=0)) for label in regret.keys()}

        # action history plots

        # f,ax = plt.subplots(3,2, figsize=(20,30))

        # for ax_, label in zip(f.axes, a_hists.keys()):
        #     bins = np.arange(n_arms+1) - 0.5
        #     ax_.hist(a_hists[label].flatten(), bins=bins)
        #     ax_.set_xticks(range(n_arms))
        #     ax_.set_xlim(-1, n_arms)
        #     ax_.set_title(label)
        
        # tikz.save(out_folder + f"tex/{simulation_id}_{testcase_id}_a_hist.tex")
        # plt.savefig(out_folder + f"png/{simulation_id}_{testcase_id}_a_hist.png")

        # f,ax = plt.subplots(3,2, figsize=(20,30))

        # for ax_, label in zip(f.axes, a_hists.keys()):
        #     bins = np.arange(n_arms+1) - 0.5
        #     ax_.plot(a_hists[label][-1,:])
        #     ax_.set_title(label)
        
        # tikz.save(out_folder + f"tex/{simulation_id}_{testcase_id}_a_hist_temp.tex")
        # plt.savefig(out_folder + f"png/{simulation_id}_{testcase_id}_a_hist_temp.png")
        # # plt.show()

    out_file = open(out_folder+f"logs_exp.json", "w")    
    json.dump(final_logs, out_file, indent = 4)
    out_file.close()
