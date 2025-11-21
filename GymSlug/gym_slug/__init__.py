from gym.envs.registration import register 

register(
	id='slug-v0',
	entry_point='gym_slug.envs:UnbreakableSeaweed',) 
	# original biomechanics 2020

register(
	id='slug-v1',
	entry_point='gym_slug.envs:BreakableSeaweed',)
	# original biomechanics 2020

register(
	id='slug-v2',
	entry_point='gym_slug.envs:UnbreakableSeaweed_Reverse',)
	# This is the environment used for the neural model (GymSlug 2.0)

register(
	id='slug-v3',
	entry_point='gym_slug.envs:UnbreakableSeaweedB38',) 
	# This is the environment used for biomechanics with mechanical coupling

register(
	id='slug-v4',
	entry_point='gym_slug.envs:UnbreakableSeaweedB38_FrictionSweep',) 
	# This is the environment used for biomechanics with added friction value fr_lumen AND mechanical coupling

register(
	id='slug-v5',
	entry_point='gym_slug.envs:UnbreakableSeaweed_FrictionSweep',) 
	# This is the environment used for biomechanics with added friction value fr_lumen
    
register(
	id='slug-v6',
	entry_point='gym_slug.envs:UnbreakableSeaweed_cont',) 
	# This is the environment used to make the biomechanics 2020 action variable continous