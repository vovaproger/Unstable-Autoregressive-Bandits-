import os
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
    out_folder = '/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/output/k_analysis_exp/'
    try:
        os.mkdir(out_folder)
        os.mkdir(out_folder+'png/')
        os.mkdir(out_folder+'tex/')
    except:
        pass

    f = open(f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/input/testcase_k_analysis_exp.json')
    param_dict = json.load(f)

    print(f'Parameters: {param_dict}')

    param_dict['gamma'] = np.array(param_dict['gamma'])

    k_values = [0, 1, 2, 4, 8, 16] # 
    param_dict['X0'] = [0]*max(k_values)
    T = param_dict['T']+len(param_dict['X0'])
    k_true = param_dict['gamma'].shape[1]-1
    n_arms = param_dict['gamma'].shape[0]
    a_hists = {}
    # Clairvoyant
    print('Training Clairvoyant algorithm')
    clrv = 'Clairvoyant'
    env = AutoregressiveEnvironment(
        n_rounds=T, gamma=param_dict['gamma'], k=k_true, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
    agent = AutoregressiveClairvoyant(
        n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k_true)
    core = Core(env, agent)
    clairvoyant_logs, a_hists['Clairvoyant'] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
    clairvoyant_logs = clairvoyant_logs[:, len(param_dict['X0']):] 

    arb_logs = {}
    arb = {}
    regret = {}
    for k in k_values:
        # ARB
        print(f'Training ARB Algorithm with k={k}')
        arb[k] = f'ARB_{k}'
        env = AutoregressiveEnvironment(
            n_rounds=T, gamma=param_dict['gamma'], k=k_true, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=param_dict['m'],
                                         sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        arb_logs[k], a_hists[k] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
        arb_logs[k] = arb_logs[k][:, len(param_dict['X0']):]
        
        regret[arb[k]] = np.inf * np.ones((param_dict['n_epochs'], T))
        for i in range(param_dict['n_epochs']):
            regret[arb[k]][i, :] = clairvoyant_logs[:100, :50016][i, :] - arb_logs[k][:100, :50016][i, :]

    #creating a graph
    sqrtn = np.sqrt(param_dict['n_epochs'])
    f, ax = plt.subplots(1, figsize=(20, 30))
    x = np.arange(len(param_dict['X0']),T, step=500)
    for k in k_values:
        ax.plot(x, np.mean(np.cumsum(regret[arb[k]].T, axis=0), axis=1)[x],
                label=arb[k])
        ax.fill_between(x, np.mean(np.cumsum(regret[arb[k]].T, axis=0), axis=1)[x]-np.std(np.cumsum(regret[arb[k]].T, axis=0), axis=1)[x]/sqrtn,
                        np.mean(np.cumsum(regret[arb[k]].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[arb[k]].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3)

        ax.set_xlim(left=0)
        ax.set_title('Cumulative Regret')
        ax.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)

    tikz.save(out_folder+f"tex/testcase_k_analysis.tex")
    plt.savefig(out_folder+f"png/testcase_k_analysis.png")

