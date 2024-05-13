import os
import csv
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

    for testcase_id in sys.argv[2:]:
        # Gamma=original
        print(f'################## Experimental Testcase {testcase_id} ###################')

        f = open(f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/input/{simulation_id}_{testcase_id}.json')
        param_dict = json.load(f)

        #extracting Gamma from the testcase_id

        input_testcase = str(testcase_id)

        testcase = input_testcase.split('=', 1)[0]

        print('Gamma=original')

        print(f'Initial Parameters: {param_dict}')

        logs = {}
        a_hists = {}

        param_dict['gamma']=np.array(param_dict['gamma'])

        T = param_dict['T']
        k = param_dict['gamma'].shape[1]-1
        n_arms = param_dict['gamma'].shape[0]

        #setting up the graph bounds

        top_bound = 0

        width=4.5
        height=4.5

        if testcase == "Gamma6":
            top_bound=50 # 10000
        elif testcase == "Gamma8":
            top_bound=5e6
        elif testcase == "Gamma11":
            top_bound=1e7
        else:
            top_bound=100000

        #calculating Gamma

        gammas=[]

        gamma_array=[]

        for key in param_dict['gamma']:
            gamma_array = np.append(gamma_array,np.cumsum(key)[-1] - key[0])

        Gamma1=np.round(gamma_array.max(),3)

        gammas=np.append(gammas,Gamma1)

        # Clairvoyant
        print('Training Clairvoyant algorithm for '+"\u0393"+"="+str(Gamma1))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, 
                    gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_orig, a_hists['Clairvoyant'+"\u0393"+"="+str(Gamma1)] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_orig = clairvoyant_logs_orig[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_orig.max()

         # ARB
        print('Training ARB Algorithm for '+"\u0393"+"="+str(Gamma1))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+"\u0393"+"="+str(Gamma1)], a_hists[testcase_id+"\u0393"+"="+str(Gamma1)] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+"\u0393"+"="+str(Gamma1)] = logs[testcase_id+"\u0393"+"="+str(Gamma1)][:, len(param_dict['X0']):]

        # Gamma=0.95

        f = open(f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/input/{simulation_id}_{testcase}=095.json')
        param_dict = json.load(f)

        param_dict['gamma']=np.array(param_dict['gamma'])

        print("Gamma=0.95")

        # T = param_dict['T']
        # k = param_dict['gamma'].shape[1]-1
        # n_arms = param_dict['gamma'].shape[0]

        gamma_array=[]

        for key in param_dict['gamma']:
            gamma_array = np.append(gamma_array,np.cumsum(key)[-1] - key[0])

        Gamma2=np.round(gamma_array.max(),3)

        gammas=np.append(gammas,Gamma2)

        # Clairvoyant
        print('Training Clairvoyant algorithm for '+"\u0393"+"="+str(Gamma2))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, 
                    gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_095, a_hists['Clairvoyant'+"\u0393"+"="+str(Gamma2)] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_095 = clairvoyant_logs_095[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_095.max()

         # ARB
        print('Training ARB Algorithm for '+"\u0393"+"="+str(Gamma2))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+"\u0393"+"="+str(Gamma2)], a_hists[testcase_id+"\u0393"+"="+str(Gamma2)] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+"\u0393"+"="+str(Gamma2)] = logs[testcase_id+"\u0393"+"="+str(Gamma2)][:, len(param_dict['X0']):]

        # Gamma=0.98
        
        f = open(f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/input/{simulation_id}_{testcase}=098.json')
        param_dict = json.load(f)

        param_dict['gamma']=np.array(param_dict['gamma'])

        print("Gamma=0.98")

        # T = param_dict['T']
        # k = param_dict['gamma'].shape[1]-1
        # n_arms = param_dict['gamma'].shape[0]

        gamma_array=[]

        for key in param_dict['gamma']:
            gamma_array = np.append(gamma_array,np.cumsum(key)[-1] - key[0])

        Gamma3=np.round(gamma_array.max(),3)

        gammas=np.append(gammas,Gamma3)

        # Clairvoyant
        print('Training Clairvoyant algorithm for '+"\u0393"+"="+str(Gamma3))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, 
                    gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_098, a_hists['Clairvoyant'+"\u0393"+"="+str(Gamma3)] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_098 = clairvoyant_logs_098[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_098.max()

         # ARB
        print('Training ARB Algorithm for '+"\u0393"+"="+str(Gamma3))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+"\u0393"+"="+str(Gamma3)], a_hists[testcase_id+"\u0393"+"="+str(Gamma3)] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+"\u0393"+"="+str(Gamma3)] = logs[testcase_id+"\u0393"+"="+str(Gamma3)][:, len(param_dict['X0']):]


        # Gamma=0.999

        f = open(f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/input/{simulation_id}_{testcase}=0999.json')
        param_dict = json.load(f)

        param_dict['gamma']=np.array(param_dict['gamma'])

        print("Gamma=0.999")

        # T = param_dict['T']
        # k = param_dict['gamma'].shape[1]-1
        # n_arms = param_dict['gamma'].shape[0]

        gamma_array=[]

        for key in param_dict['gamma']:
            gamma_array = np.append(gamma_array,np.cumsum(key)[-1] - key[0])

        Gamma4=np.round(gamma_array.max(),3)

        gammas=np.append(gammas,Gamma4)

        # Clairvoyant
        print('Training Clairvoyant algorithm for '+"\u0393"+"="+str(Gamma4))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, 
                    gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_0999, a_hists['Clairvoyant'+"\u0393"+"="+str(Gamma4)] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_0999 = clairvoyant_logs_0999[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_0999.max()

         # ARB
        print('Training ARB Algorithm for '+"\u0393"+"="+str(Gamma4))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+"\u0393"+"="+str(Gamma4)], a_hists[testcase_id+"\u0393"+"="+str(Gamma4)] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+"\u0393"+"="+str(Gamma4)] = logs[testcase_id+"\u0393"+"="+str(Gamma4)][:, len(param_dict['X0']):]

        f = open(f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/input/{simulation_id}_{testcase}=1.json')
        param_dict = json.load(f)

        param_dict['gamma']=np.array(param_dict['gamma'])

        print("Gamma=1")

        gamma_array=[]

        for key in param_dict['gamma']:
            gamma_array = np.append(gamma_array,np.cumsum(key)[-1] - key[0])

        Gamma5=np.round(gamma_array.max(),3)

        gammas=np.append(gammas,Gamma5)

        # Clairvoyant
        print('Training Clairvoyant algorithm for '+"\u0393"+"="+str(Gamma5))
        clrv = 'Clairvoyant'
        env = AutoregressiveEnvironment(n_rounds=T, 
                    gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveClairvoyant(n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
        core = Core(env, agent)
        clairvoyant_logs_1, a_hists['Clairvoyant'+"\u0393"+"="+str(Gamma5)] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        clairvoyant_logs_1 = clairvoyant_logs_1[:, len(param_dict['X0']):]

        # Reward upper bound 
        max_reward = clairvoyant_logs_1.max()

         # ARB
        print('Training ARB Algorithm for '+"\u0393"+"="+str(Gamma5))
        env = AutoregressiveEnvironment(n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'], sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        logs[testcase_id+"\u0393"+"="+str(Gamma5)], a_hists[testcase_id+"\u0393"+"="+str(Gamma5)] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        logs[testcase_id+"\u0393"+"="+str(Gamma5)] = logs[testcase_id+"\u0393"+"="+str(Gamma5)][:, len(param_dict['X0']):]

        # Regrets computing
        print('Computing regrets...')
        regret = {label : np.inf * np.ones((param_dict['n_epochs'], T)) for label in logs.keys()}
        for i in range(param_dict['n_epochs']):
            regret[testcase_id+"\u0393"+"="+str(Gamma1)][i, :] = clairvoyant_logs_orig[i, :] - logs[testcase_id+"\u0393"+"="+str(Gamma1)][i, :]
            regret[testcase_id+"\u0393"+"="+str(Gamma2)][i, :] = clairvoyant_logs_095[i, :] - logs[testcase_id+"\u0393"+"="+str(Gamma2)][i, :]
            regret[testcase_id+"\u0393"+"="+str(Gamma3)][i, :] = clairvoyant_logs_098[i, :] - logs[testcase_id+"\u0393"+"="+str(Gamma3)][i, :]
            regret[testcase_id+"\u0393"+"="+str(Gamma4)][i, :] = clairvoyant_logs_0999[i, :] - logs[testcase_id+"\u0393"+"="+str(Gamma4)][i, :]
            regret[testcase_id+"\u0393"+"="+str(Gamma5)][i, :] = clairvoyant_logs_1[i, :] - logs[testcase_id+"\u0393"+"="+str(Gamma5)][i, :]
    
        print('Creating Graphs...')

        #  cumulative regret plot - replace f and ax with f_reg and ax_reg
        
        f_reg,ax_reg = plt.subplots(1, figsize=(width,height))
        x = np.arange(len(param_dict['X0']),T+50, step=50)
        x[-1] = min(x[-1],len(np.mean(np.cumsum(regret[testcase_id+"\u0393"+"="+str(Gamma2)].T, axis=0), axis=1))-1)
        sqrtn = np.sqrt(param_dict['n_epochs'])

        for i,label in enumerate(regret.keys()):
            if label==testcase_id+"\u0393"+"="+str(Gamma5):
                ax_reg.plot(x, np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x], dashes=[6, 2],label="\u0393"+"="+str(gammas[i]), color=f'C{i+2}')
            else:
                ax_reg.plot(x, np.log(np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]), label="\u0393"+"="+str(gammas[i]), color=f'C{i+2}')
            ax_reg.fill_between(x, np.log(np.mean(np.cumsum(regret[label].T, axis=0),axis=1)[x]-np.std(np.cumsum(regret[label].T, axis=0),axis=1)[x]/sqrtn),
                        np.log(np.mean(np.cumsum(regret[label].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[label].T, axis=0), axis=1)[x]/sqrtn), alpha=0.3, color=f'C{i+2}')
            plt.subplots_adjust(left=0.15)
            ax_reg.set_xlim(left=0)
            ax_reg.set_ylim(bottom=0, top=top_bound)
            ax_reg.set_title('ARB-Cumulative Regret')
            ax_reg.legend()    

        tikz.save(out_folder + f"tex/{simulation_id}_{testcase_id}_regret_2_log.tex")
        plt.savefig(out_folder + f"png/{simulation_id}_{testcase_id}_regret_2_log.png")

        # logging 

        # final_logs[f'testcase_{testcase_id}'] = {label : np.mean(np.sum(regret[label].T, axis=0)) for label in regret.keys()}

        # # action history plots

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
