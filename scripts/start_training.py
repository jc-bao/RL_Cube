#!/usr/bin/env python
# training script
import gym
import time
import numpy
import random
import time
import qlearn
from gym import wrappers

# ROS packages required
import rospy
import rospkg
from functools import reduce
# import our training environment
import my_one_disk_walk


if __name__ == '__main__':
    
    rospy.init_node('cartpole_gym', anonymous=True, log_level=rospy.WARN)
    # Create the Gym environment
    # TODO change name
    env = gym.make('MyMovingCubeOneDiskWalkEnv-v0')
    rospy.loginfo ( "Gym environment done")
        
    # Set the logging system
    # TODO change name
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_moving_cube_pkg')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True) 
    rospy.loginfo ( "Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    # TODO load parameter
    Alpha = rospy.get_param("/moving_cube/alpha")
    Epsilon = rospy.get_param("/moving_cube/epsilon")
    Gamma = rospy.get_param("/moving_cube/gamma")
    epsilon_discount = rospy.get_param("/moving_cube/epsilon_discount")
    nepisodes = rospy.get_param("/moving_cube/nepisodes")
    nsteps = rospy.get_param("/moving_cube/nsteps")

    running_step = rospy.get_param("/moving_cube/running_step")

    # Initialises the algorithm that we are going to use for learning
    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE => " + str(x))
        
        cumulated_reward = 0  
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        
        # Initialize the environment and get first state of the robot
        
        observation = env.reset()
        state = ''.join(map(str, observation))
        
        # Show on screen the actual situation of the robot
        # for each episode, we test the robot for nsteps
        for i in range(nsteps):
            
            rospy.loginfo("############### Start Step => "+str(i))
            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            rospy.loginfo ("Next action is: %d", action)
            # Execute the action in the environment and get feedback
            observation, reward, done, info = env.step(action)
            rospy.loginfo(str(observation) + " " + str(reward))
            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            # Make the algorithm learn based on the results
            #rospy.logwarn("############### State we were => " + str(state))
            #rospy.logwarn("############### Action that we took => " + str(action))
            #rospy.logwarn("############### Reward that action gave => " + str(reward))
            #rospy.logwarn("############### State in which we will start next step => " + str(nextState))
            qlearn.learn(state, action, reward, nextState)

            if not(done):
                state = nextState
            else:
                rospy.loginfo ("DONE")
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break
            rospy.loginfo("############### END Step =>" + str(i))
            #raw_input("Next Step...PRESS KEY")
            #rospy.sleep(2.0)
        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.logwarn ( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))

    
    rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    # TODO define where to store
    '''
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('my_cube_training')
    outdir = pkg_path + '/training_results'
        
    # Check if it already exists a training policy, and load it if so
    qfile = os.path.join(outdir,"qlearn_states.npy")
    # qfile = "qlearn_states.npy"
    if (os.path.exists(qfile)):
        print("Loading from file:",qfile)
        qlearn.load(qfile)
    else:
        print("The File doesnt exist="+str(qfile))
    '''
    #Save the trained policy
    qlearn.save(qfile)

    env.close()
