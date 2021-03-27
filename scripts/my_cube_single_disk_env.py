import rospy
import numpy
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from openai_ros import robot_gazebo_env


class MyCubeSingleDiskEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self, init_roll_vel):
        """Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.
        self.init_roll_vel = init_roll_vel
        # Internal Vars
        self.controllers_list = ['joint_state_controller','inertia_wheel_roll_joint_velocity_controller']

        self.robot_name_space = "moving_cube"

        reset_controls_bool = True
        
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        
        super(MyRobotEnv, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=reset_controls_bool)

        """
        To check any topic we need to have the simulations running, we need to do two things:
        1) Unpause the simulation: without that the stream of data doesn't flow. This is for simulations
        that are pause for whatever reason
        2) If the simulation was running already for some reason, we need to reset the controllers.
        This has to do with the fact that some plugins with tf don't understand the reset of the simulation
        and need to be reset to work properly.
        """
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()

        # We Start all the ROS related Subscribers and publishers
        rospy.Subscriber("/moving_cube/joint_states", JointState, self._joints_callback)
        rospy.Subscriber("/moving_cube/odom", Odometry, self._odom_callback)

        self._roll_vel_pub = rospy.Publisher('/moving_cube/inertia_wheel_roll_joint_velocity_controller/command',
                                             Float64, queue_size=1)

        self._check_publishers_connection()

        self.gazebo.pauseSim()
    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        self._check_publishers_connection()
        return True
    
    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()
        self._check_odom_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/moving_cube/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current moving_cube/joint_states READY=>" + str(self.joints))

            except:
                rospy.logerr("Current moving_cube/joint_states not ready yet, retrying for getting joint_states")
        return self.joints

    def _check_odom_ready(self):
        self.odom = None
        while self.odom is None and not rospy.is_shutdown():
            try:
                self.odom = rospy.wait_for_message("/moving_cube/odom", Odometry, timeout=1.0)
                rospy.logdebug("Current /moving_cube/odom READY=>" + str(self.odom))

            except:
                rospy.logerr("Current /moving_cube/odom not ready yet, retrying for getting odom")

        return self.odom

    def _check_publishers_connection(self):
        """
        Checks that all the publishers are working
        :return:
        """
        rate = rospy.Rate(10)  # 10hz
        while self._roll_vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.logdebug("No susbribers to _roll_vel_pub yet so we wait and try again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                # This is to avoid error when world is rested, time when backwards.
                pass
        rospy.logdebug("_roll_vel_pub Publisher Connected")

        rospy.logdebug("All Publishers READY")
    
    # CubeSingleDiskEnv virtual methods
    # ----------------------------
    
    def _joints_callback(self, data):
        self.joints = data
    
    def _odom_callback(self, data):
        self.odom = data

    # Methods that the TaskEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TaskEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TaskEnvironment will need.
    # ----------------------------

    def move_joints(self, roll_speed):
        joint_speed_value = Float64()
        joint_speed_value.data = roll_speed
        rospy.logdebug("Single Disk Roll Velocity>>" + str(joint_speed_value))
        self._roll_vel_pub.publish(joint_speed_value)
        self.wait_until_roll_is_in_vel(joint_speed_value.data)
    
    def wait_until_roll_is_in_vel(self, velocity):
        """
        This method will wait until the roll disk wheel reaches the desired speed, with a certain error.
        """
        rate = rospy.Rate(10)
        start_wait_time = rospy.get_rostime().to_sec()
        end_wait_time = 0.0
        epsilon = 0.1
        v_plus = velocity + epsilon
        v_minus = velocity - epsilon
        while not rospy.is_shutdown():
            joint_data = self._check_joint_states_ready()
            roll_vel = joint_data.velocity[0]
            rospy.logdebug("VEL=" + str(roll_vel) + ", ?RANGE=[" + str(v_minus) + ","+str(v_plus)+"]")
            are_close = (roll_vel <= v_plus) and (roll_vel > v_minus)
            if are_close:
                rospy.logdebug("Reached Velocity!")
                end_wait_time = rospy.get_rostime().to_sec()
                break
            rospy.logdebug("Not there yet, keep waiting...")
            rate.sleep()
        delta_time = end_wait_time- start_wait_time
        rospy.logdebug("[Wait Time=" + str(delta_time)+"]")
        return delta_time

    def get_joints(self):
    return self.joints

    def get_odom(self):
        return self.odom