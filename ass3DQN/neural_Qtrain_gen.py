import sys
import gym
import tensorflow as tf
import numpy as np
import random
import datetime
import heapq

"""
Hyper Parameters
"""
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 0.6  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
EPSILON_DECAY_STEPS = 100
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 128  # size of minibatch
TEST_FREQUENCY = 10  # How many episodes to run before visualizing test accuracy
SAVE_FREQUENCY = 1000  # How many episodes to run before saving model (unused)
NUM_EPISODES = 100  # Episode limitation
# 200
EP_MAX_STEPS = 200  # Step limitation in an episode
# The number of test iters (with epsilon set to 0) to run every TEST_FREQUENCY episodes
NUM_TEST_EPS = 4
HIDDEN_NODES = 5

# used to identify environment
# ENVIRONMENT_NAME = None


def init(env, env_name):
    """
    Initialise any globals, e.g. the replay_buffer, epsilon, etc.
    return:
        state_dim: The length of the state vector for the env
        action_dim: The length of the action space, i.e. the number of actions

    NB: for discrete action envs such as the cartpole and mountain car, this
    function can be left unchanged.

    Hints for envs with continuous action spaces, e.g. "Pendulum-v0"
    1) you'll need to modify this function to discretise the action space and
    create a global dictionary mapping from action index to action (which you
    can use in `get_env_action()`)
    2) for Pendulum-v0 `env.action_space.low[0]` and `env.action_space.high[0]`
    are the limits of the action space.
    3) setting a global flag iscontinuous which you can use in `get_env_action()`
    might help in using the same code for discrete and (discretised) continuous
    action spaces
    """
    global replay_buffer, epsilon
    replay_buffer = []
    epsilon = INITIAL_EPSILON

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    return state_dim, action_dim


def get_network(state_dim, action_dim, hidden_nodes=HIDDEN_NODES):
    """Define the neural network used to approximate the q-function

    The suggested structure is to have each output node represent a Q value for
    one action. e.g. for cartpole there will be two output nodes.

    Hints:
    1) Given how q-values are used within RL, is it necessary to have output
    activation functions?
    2) You will set `target_in` in `get_train_batch` further down. Probably best
    to implement that before implementing the loss (there are further hints there)
    """
    state_in = tf.placeholder("float", [None, state_dim])
    action_in = tf.placeholder("float", [None, action_dim])  # one hot

    # used for target net
    global next_state_in
    next_state_in = tf.placeholder("float", [None, state_dim])

    # q value for the target network for the state, action taken
    target_in = tf.placeholder("float", [None])

    # TO IMPLEMENT: Q network, whose input is state_in, and has action_dim outputs
    # which are the network's esitmation of the Q values for those actions and the
    # input state. The final layer should be assigned to the variable q_values
    # ...
    # tf.contrib.layers.xavier_initializer(uniform=True)
    hidden_nodes = 20
    # hidden_nodes_2 = 64

    w_initializer, b_initializer = \
        tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1) 

    # --------- eval net
    with tf.variable_scope('eval_net'):
        c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope('l1'):
            w1 = tf.Variable(tf.random_normal([state_dim, hidden_nodes], name = 'w1'), collections = c_names)
            b1 = tf.Variable(tf.random_normal([hidden_nodes], name = 'b1'), collections = c_names)

        with tf.variable_scope('hidden'):
            w_hidden = tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes], name = 'w_hidden'), collections = c_names)
            b_hidden = tf.Variable(tf.random_normal([hidden_nodes], name = 'b_hidden'), collections = c_names)

        with tf.variable_scope('hidden_2'):
            w_hidden_2 = tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes], name = 'w_hidden_2'), collections = c_names)
            b_hidden_2 = tf.Variable(tf.random_normal([hidden_nodes], name = 'b_hidden_2'), collections = c_names)

        with tf.variable_scope('l2'):
            w2 = tf.Variable(tf.random_normal([hidden_nodes, action_dim], name = 'w2'), collections = c_names)
            b2 = tf.Variable(tf.random_normal([action_dim], name = 'b2'), collections = c_names)
        
    l1 = tf.nn.relu(tf.matmul(state_in, w1) + b1)
    l_hidden = tf.nn.relu(tf.matmul(l1, w_hidden) + b_hidden)
    l_hidden_2 = tf.nn.relu(tf.matmul(l_hidden, w_hidden_2) + b_hidden_2)
    q_values = tf.matmul(l_hidden_2, w2) + b2

    # -------------- define tranining steps for q_values
    q_selected_action = \
        tf.reduce_sum(tf.multiply(q_values, action_in), reduction_indices=1)

    # TO IMPLEMENT: loss function
    # should only be one line, if target_in is implemented correctly
    loss = tf.reduce_mean(tf.squared_difference(target_in, q_selected_action))
    # regularization for weights
    regularization_parameter = 0.001
    for w in [w1, w_hidden, w_hidden_2, w2]:
        loss += regularization_parameter * tf.reduce_sum(tf.square(w))
    
    optimise_step = tf.train.AdamOptimizer().minimize(loss)

    train_loss_summary_op = tf.summary.scalar("TrainingLoss", loss)

    # ---------- target net(delayed updates)
    global q_target
    with tf.variable_scope('target_net'):
        c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        with tf.variable_scope('l1'):
            w1 = tf.Variable(tf.random_normal([state_dim, hidden_nodes], name = 'w1'), collections = c_names)
            b1 = tf.Variable(tf.random_normal([hidden_nodes], name = 'b1'), collections = c_names)

        with tf.variable_scope('hidden'):
            w_hidden = tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes], name = 'w_hidden'), collections = c_names)
            b_hidden = tf.Variable(tf.random_normal([hidden_nodes], name = 'b_hidden'), collections = c_names)

        with tf.variable_scope('hidden_2'):
            w_hidden_2 = tf.Variable(tf.random_normal([hidden_nodes, hidden_nodes], name = 'w_hidden_2'), collections = c_names)
            b_hidden_2 = tf.Variable(tf.random_normal([hidden_nodes], name = 'b_hidden_2'), collections = c_names)

        with tf.variable_scope('l2'):
            w2 = tf.Variable(tf.random_normal([hidden_nodes, action_dim], name = 'w2'), collections = c_names)
            b2 = tf.Variable(tf.random_normal([action_dim], name = 'b2'), collections = c_names)

    l1 = tf.nn.relu(tf.matmul(next_state_in, w1) + b1)
    l_hidden = tf.nn.relu(tf.matmul(l1, w_hidden) + b_hidden)
    l_hidden_2 = tf.nn.relu(tf.matmul(l_hidden, w_hidden_2) + b_hidden_2)
    q_target = tf.matmul(l_hidden_2, w2) + b2

    # --------- operation to update params of target_net
    global replace_target_param_op
    t_params = tf.get_collection('target_net_params')
    e_params = tf.get_collection('eval_net_params')
    replace_target_param_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
    
    return state_in, action_in, target_in, q_values, q_selected_action, \
           loss, optimise_step, train_loss_summary_op


def init_session():
    global session, writer
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())

    # Setup Logging
    logdir = "tensorboard/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, session.graph)


def get_action(state, state_in, q_values, epsilon, test_mode, action_dim):
    Q_estimates = q_values.eval(feed_dict={state_in: [state]})[0]
    epsilon_to_use = 0.0 if test_mode else epsilon
    if random.random() < epsilon_to_use:
        action = random.randint(0, action_dim - 1)
    else:
        action = np.argmax(Q_estimates)
    return action


def get_env_action(action):
    """
    Modify for continous action spaces that you have discretised, see hints in
    `init()`
    """
    return action


def update_replay_buffer(replay_buffer, state, action, reward, next_state, done,
                         action_dim,
                         state_in, action_in, target_in, q_values, q_selected_action):
    """
    Update the replay buffer with provided input in the form:
    (state, one_hot_action, reward, next_state, done)

    Hint: the minibatch passed to do_train_step is one entry (randomly sampled)
    from the replay_buffer
    """
    # TO IMPLEMENT: append to the replay_buffer
    # ensure the action is encoded one hot
    # ...
    # append to buffer
    # one_hot_action = tf.one_hot([action], action_dim)
    # print(one_hot_action.eval()[0])

    # if not done:
    #     reward = GAMMA *reward
    # if done:
    #     reward = -50


    ###### now take replay_buffer as heapq

    one_hot_action = np.zeros(action_dim)
    one_hot_action[action] = 1  

    global q_target
    global next_state_in
    Q_batch_target_next_state = q_target.eval(feed_dict={
        next_state_in: [next_state]
    })
    Q_batch_eval_next_state = q_values.eval(feed_dict={
        state_in: [next_state]
    })
    # Q_batch_eval_current_state = q_values.eval(feed_dict={
    #     state_in: [state]
    # })
    Q_selected_action_val = q_selected_action.eval(feed_dict={
        action_in: [one_hot_action],
        state_in: [state]
    })

    max_act_for_next = np.argmax(Q_batch_eval_next_state, axis=1)
    Q_value = Q_batch_target_next_state[0][max_act_for_next[0]]
    target_val = reward + GAMMA * Q_value

    priority = np.abs(Q_selected_action_val[0] - target_val)

    heapq.heappush(replay_buffer,(priority, state, one_hot_action, reward, next_state, done))

    # replay_buffer.append((state, one_hot_action, reward, next_state, done))
    # Ensure replay_buffer doesn't grow larger than REPLAY_SIZE
    if len(replay_buffer) > REPLAY_SIZE:
        try:
            heapq.heappop(replay_buffer)
        except ValueError:
            replay_buffer.pop(0)
    return None


def do_train_step(replay_buffer, state_in, action_in, target_in,
                  q_values, q_selected_action, loss, optimise_step,
                  train_loss_summary_op, batch_presentations_count):
    target_batch, state_batch, action_batch = \
        get_train_batch(q_values, state_in, replay_buffer)

    summary,  _ = session.run([train_loss_summary_op, optimise_step], feed_dict={
        target_in: target_batch,
        state_in: state_batch,
        action_in: action_batch
    })

    writer.add_summary(summary, batch_presentations_count)


def get_train_batch(q_values, state_in, replay_buffer):
    """
    Generate Batch samples for training by sampling the replay buffer"
    Batches values are suggested to be the following;
        state_batch: Batch of state values
        action_batch: Batch of action values
        target_batch: Target batch for (s,a) pair i.e. one application
            of the bellman update rule.

    return:
        target_batch, state_batch, action_batch

    Hints:
    1) To calculate the target batch values, you will need to use the
    q_values for the next_state for each entry in the batch.
    2) The target value, combined with your loss defined in `get_network()` should
    reflect the equation in the middle of slide 12 of Deep RL 1 Lecture
    notes here: https://webcms3.cse.unsw.edu.au/COMP9444/17s2/resources/12494
    """
    # minibatch = random.sample(replay_buffer, BATCH_SIZE)

    minibatch = []
    global ENVIRONMENT_NAME
    high_priority_example_ratio = 0
    low_priority_example_ratio = 0
    if ENVIRONMENT_NAME == 'MountainCar-v0':
        # 80% of minibatch will be filled with data from the top 20% of replay_buffer
        high_priority_example_ratio = 0.8
        low_priority_example_ratio = 0.2
    else:
        high_priority_example_ratio = 0.6
        low_priority_example_ratio = 0.4

    if int(len(replay_buffer) * low_priority_example_ratio) < int(BATCH_SIZE * high_priority_example_ratio):
        temp_high_priority_entries = heapq.nlargest(int(BATCH_SIZE * high_priority_example_ratio), replay_buffer)
        for e in temp_high_priority_entries:
            minibatch.append(e[1:])

        temp_normal_priority_entries = random.sample(replay_buffer[:-int(BATCH_SIZE * high_priority_example_ratio)], BATCH_SIZE - int(BATCH_SIZE * high_priority_example_ratio))
        for e in temp_normal_priority_entries:
            minibatch.append(e[1:])
    else:
        temp_high_priority_entries = random.sample(replay_buffer[-(int(len(minibatch) * low_priority_example_ratio)):], int(BATCH_SIZE * high_priority_example_ratio))
        for e in temp_high_priority_entries:
            minibatch.append(e[1:])

        temp_normal_priority_entries = random.sample(replay_buffer[:-(int(len(minibatch) * low_priority_example_ratio))], BATCH_SIZE - int(BATCH_SIZE * high_priority_example_ratio))
        for e in temp_normal_priority_entries:
            minibatch.append(e[1:])

    if (len(minibatch) != BATCH_SIZE):
        print(len(minibatch))
        sys.exit()

    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    target_batch = []
    # instead of using q_values.eval(...), now to target_net for target value
    global q_target
    global next_state_in
    Q_batch_target_next_state = q_target.eval(feed_dict={
        next_state_in: next_state_batch
    })
    Q_batch_eval_next_state = q_values.eval(feed_dict={
        state_in: next_state_batch
    })
    # Q_batch_eval_current_state = q_values.eval(feed_dict={
    #     state_in: state_batch
    # })

    max_act_for_next = np.argmax(Q_batch_eval_next_state, axis=1)
    batch_index = np.arange(BATCH_SIZE, dtype=np.int32)
    Q_value_batch = Q_batch_target_next_state[batch_index, max_act_for_next]
    # print(Q_value_batch)

    for i in range(0, BATCH_SIZE):
        sample_is_done = minibatch[i][4]
        if sample_is_done:
            target_batch.append(reward_batch[i])
        else:
            # TO IMPLEMENT: set the target_val to the correct Q value update
            # target_val = reward_batch[i] + GAMMA * np.max(Q_value_batch[i])
            target_val = reward_batch[i] + GAMMA * Q_value_batch[i]
            # print('target_val:', target_val)
            target_batch.append(target_val)
    return target_batch, state_batch, action_batch

def qtrain(env, state_dim, action_dim,
           state_in, action_in, target_in, q_values, q_selected_action,
           loss, optimise_step, train_loss_summary_op,
           num_episodes=NUM_EPISODES, ep_max_steps=EP_MAX_STEPS,
           test_frequency=TEST_FREQUENCY, num_test_eps=NUM_TEST_EPS,
           final_epsilon=FINAL_EPSILON, epsilon_decay_steps=EPSILON_DECAY_STEPS,
           force_test_mode=False, render=True):

    # init, set target_net same as eval_net
    global replace_target_param_op
    session.run(replace_target_param_op)

    global ENVIRONMENT_NAME
    global epsilon
    # Record the number of times we do a training batch, take a step, and
    # the total_reward across all eps
    batch_presentations_count = total_steps = total_reward = 0

    record_last_hundred_reward = []

    num_episodes = 1000
    collect_first_positive_test_result_for_mountain_car = False
    for episode in range(num_episodes):
        # initialize task
        state = env.reset()
        if render: env.render()

        # Update epsilon once per episode - exp decaying
        epsilon -= (epsilon - final_epsilon) / epsilon_decay_steps

        # in test mode we set epsilon to 0
        test_mode = force_test_mode or \
                    ((episode % test_frequency) < num_test_eps and
                        episode > num_test_eps
                    )
        if test_mode: print("Test mode (epsilon set to 0.0)")

        ep_reward = 0
        # for step in range(ep_max_steps):
        step = 0
        
        while step <= ep_max_steps:
            step += 1

            total_steps += 1

            # get an action and take a step in the environment
            if ENVIRONMENT_NAME == 'MountainCar-v0' and collect_first_positive_test_result_for_mountain_car == False:
                # use merely random to collect the very first success test result
                action = env.action_space.sample()
            else:
                action = get_action(state, state_in, q_values, epsilon, test_mode,
                                    action_dim)
            env_action = get_env_action(action)
            next_state, reward, done, _ = env.step(env_action)

            # specified for game mountainCar
            if ENVIRONMENT_NAME == 'MountainCar-v0':
                if next_state[0] >= 0.5:
                    reward = 100
                    done = True
                    step = ep_max_steps
                    collect_first_positive_test_result_for_mountain_car = True
                elif next_state[0] >= 0.4 and next_state[1] > 0:
                    reward = 8
                elif next_state[0] >= 0.3 and next_state[1] > 0:
                    reward = 6
                elif next_state[0] >= 0.2 and next_state[1] > 0:
                    reward = 4
                elif next_state[0] >= 0.0 and next_state[1] > 0:
                    reward = 2
                elif next_state[0] <= -1.1 and next_state[1] > 0:
                    reward = 6
                elif next_state[0] <= -1.0 and next_state[1] > 0:
                    reward = 4
                elif next_state[0] <= -0.9 and next_state[1] > 0:
                    reward = 2
                
                if collect_first_positive_test_result_for_mountain_car == False:
                    # it will keep searching until find a correct one
                    done = False
                    step = 0

            ep_reward += reward

            # display the updated environment
            if render: env.render()  # comment this line to possibly reduce training time

            # add the s,a,r,s' samples to the replay_buffer
            update_replay_buffer(replay_buffer, state, action, reward,
                                 next_state, done, action_dim,
                                 state_in, action_in, target_in, q_values, q_selected_action) # newly added for priority queue
            state = next_state

            # perform a training step if the replay_buffer has a batch worth of samples
            if (len(replay_buffer) > BATCH_SIZE):
                do_train_step(replay_buffer, state_in, action_in, target_in,
                              q_values, q_selected_action, loss, optimise_step,
                              train_loss_summary_op, batch_presentations_count)
                batch_presentations_count += 1

            if done:
                break

        if episode < 50 and episode % 2 == 0:
            session.run(replace_target_param_op)
        else:
            session.run(replace_target_param_op)

        # self added to monitor the last 100 avg reward
        record_last_hundred_reward.append(ep_reward)
        if len(record_last_hundred_reward) > 100:
            record_last_hundred_reward.pop(0)
        print('last hundred reward avg: ', np.mean(record_last_hundred_reward))

        total_reward += ep_reward
        test_or_train = "test" if test_mode else "train"
        print("end {0} episode {1}, ep reward: {2}, ave reward: {3}, \
            Batch presentations: {4}, epsilon: {5}".format(
            test_or_train, episode, ep_reward, total_reward / (episode + 1),
            batch_presentations_count, epsilon
        ))


def setup():
    # default_env_name = 'CartPole-v0'
    default_env_name = 'MountainCar-v0'
    # default_env_name = 'Pendulum-v0'
    # if env_name provided as cmd line arg, then use that
    env_name = sys.argv[1] if len(sys.argv) > 1 else default_env_name
    global ENVIRONMENT_NAME
    ENVIRONMENT_NAME = env_name

    env = gym.make(env_name)
    state_dim, action_dim = init(env, env_name)
    network_vars = get_network(state_dim, action_dim)
    init_session()
    return env, state_dim, action_dim, network_vars


def main():
    env, state_dim, action_dim, network_vars = setup()
    qtrain(env, state_dim, action_dim, *network_vars, render=True)


if __name__ == "__main__":
    main()
