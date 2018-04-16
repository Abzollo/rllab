import atexit
from queue import Empty
from multiprocessing import Process, Queue
from rllab.sampler.utils import rollout, rollout_tf
import numpy as np
import pickle 

__all__ = [
    'init_worker',
    'init_plot',
    'update_plot',
    'init_plot_tf',
    'update_plot_tf',
]

process = None
queue = None


def _worker_start():
    env = None
    policy = None
    max_length = None
    try:
        while True:
            msgs = {}
            # Only fetch the last message of each type
            while True:
                try:
                    msg = queue.get_nowait()
                    msgs[msg[0]] = msg[1:]
                except Empty:
                    break
            if 'stop' in msgs:
                break
            elif 'update' in msgs:
                env, policy = msgs['update']
                # env.start_viewer()
            elif 'demo' in msgs:
                param_values, max_length = msgs['demo']
                policy.set_param_values(param_values)
                rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
            else:
                if max_length:
                    rollout(env, policy, max_path_length=max_length, animated=True, speedup=5)
    except KeyboardInterrupt:
        pass


def _shutdown_worker():
    if process:
        queue.put(['stop'])
        queue.close()
        process.join()


def init_worker():
    # This function is called from scripts/run_experiment_lite.py
    global process, queue
    queue = Queue()
    #process = Process(target=_worker_start)
    process = Process(target=_worker_start_tf)  # TODO: check if tensorflow
    process.start()
    atexit.register(_shutdown_worker)


def init_plot(env, policy):
    queue.put(['update', env, policy])


def update_plot(policy, max_length=np.inf):
    queue.put(['demo', policy.get_param_values(), max_length])



######################################################

def init_plot_tf(env):
    queue.put(['update', env])

def update_plot_tf(env, agent, max_length=np.inf):
    """
    Get all the actions and send them to the renderer process.
    Since initial conditions might be randomized, we preserve the
    original env and send it to the renderer with its random state.
    """
    initial_env = pickle.dumps(env)
    random_state = np.random.get_state()  # To reproduce env steps in renderer

    agent_actions = []
    agent.reset()
    next_o = env.reset()
    for _ in range(max_length):
        # Perform actions and record them (tf.session runs in this step)
        a, agent_info = agent.get_action(next_o)
        agent_actions.append((a, agent_info))
        # Get next observables
        next_o, r, d, env_info = env.step(a)
        if d: break

    # Send the env, its random_state, and the actions to the renderer process
    queue.put(['demo', initial_env, random_state, agent_actions, max_length])


def _worker_start_tf():
    env = None
    agent_actions = None
    max_length = None
    try:
        while True:
            msgs = {}
            # Only fetch the last message of each type
            while True:
                try:
                    msg = queue.get_nowait()
                    msgs[msg[0]] = msg[1:]
                except Empty:
                    break
            if 'stop' in msgs:
                break
            elif 'update' in msgs:
                env, = msgs['update']
                # env.start_viewer()
            elif 'demo' in msgs:
                pickled_env, random_state, agent_actions, max_length = msgs['demo']
                env = pickle.loads(pickled_env)
                rollout_tf(env, random_state, agent_actions, max_path_length=max_length, animated=True, speedup=5)
            else:
                if max_length:
                    rollout_tf(env, random_state, agent_actions, max_path_length=max_length, animated=True, speedup=5)
    except KeyboardInterrupt:
        pass
