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

    out_folder = '/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/output/m_bound_analysis_exp/'
    try:
        os.mkdir(out_folder)
        os.mkdir(out_folder+'png/')
        os.mkdir(out_folder+'tex/')
    except:
        pass

    f = open(f'/Users/uladzimircharniauski/Documents/AR_Bandits/Unstable-Autoregressive-Bandits/autoregressive_bandits_exp/input/testcase_m_bound_analysis.json')
    param_dict = json.load(f)

    print(f'Parameters: {param_dict}')

    param_dict['gamma'] = np.array(param_dict['gamma'])

    T = param_dict['T']+len(param_dict['X0'])
    k = param_dict['gamma'].shape[1]-1
    n_arms = param_dict['gamma'].shape[0]

    m_values = [1, 10, 100, 500, 1000, 2500]

    # Clairvoyant
    print('Training Clairvoyant algorithm')
    clrv = 'Clairvoyant'
    env = AutoregressiveEnvironment(
        n_rounds=T, gamma=param_dict['gamma'], 
        k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
    agent = AutoregressiveClairvoyant(
        n_arms=n_arms, gamma=param_dict['gamma'], X0=param_dict['X0'], k=k)
    core = Core(env, agent)
    clairvoyant_logs = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)
    clairvoyant_logs = clairvoyant_logs[:, len(param_dict['X0']):]

    arb_logs = {}
    arb = {}
    regret = {}
    for m in m_values:
        # ARB
        print(f'Training ARB Algorithm with m={m}')
        arb[m] = f'ARB_{m}'
        env = AutoregressiveEnvironment(
            n_rounds=T, gamma=param_dict['gamma'], k=k, noise_std=param_dict['noise_std'], X0=param_dict['X0'], random_state=param_dict['seed'])
        agent = AutoregressiveRidgeAgent(n_arms, param_dict['X0'], k,  m=m,
                                         sigma_=param_dict['noise_std'], delta_=param_dict['delta'], lambda_=param_dict['lambda'])
        core = Core(env, agent)
        arb_logs[m] = core.simulation(n_epochs=param_dict['n_epochs'], n_rounds=T)[:, len(param_dict['X0']):]
        regret[arb[m]] = np.inf * np.ones((param_dict['n_epochs'], T))
        for i in range(param_dict['n_epochs']): 
            regret[arb[m]][i, :] = clairvoyant_logs[i, :] - arb_logs[m][i, :] # time 
    #creating a graph
    sqrtn = np.sqrt(param_dict['n_epochs'])
    f, ax = plt.subplots(1, figsize=(20, 30))
    x = np.arange(len(param_dict['X0']),T, step=50)
    for m in m_values:
        ax.plot(x, np.mean(np.cumsum(regret[arb[m]].T, axis=0), axis=1)[x],
                label=arb[m])
        ax.fill_between(x, np.mean(np.cumsum(regret[arb[m]].T, axis=0), axis=1)[x]-np.std(np.cumsum(regret[arb[m]].T, axis=0), axis=1)[x]/sqrtn,
                        np.mean(np.cumsum(regret[arb[m]].T, axis=0), axis=1)[x]+np.std(np.cumsum(regret[arb[m]].T, axis=0), axis=1)[x]/sqrtn, alpha=0.3)

        ax.set_xlim(left=0)
        ax.set_title('Cumulative Regret')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)

    plt.savefig(out_folder+f"png/testcase_m_bound_analysis_2.png")
    tikz.save(out_folder+f"tex/testcase_m_bound_analysis_2.tex")
    #plt.show()
