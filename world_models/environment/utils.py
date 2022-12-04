from gym_duckietown.simulator import Simulator
from world_models.environment.wrappers import DtRewardWrapper, DiscreteWrapper

def load_environment(
    map_name='loop_pedestrians',
    camera_width=80,
    camera_height=60,
    draw_curve=True,
    max_steps=3500,
    distortion=False,
    domain_rand=False,
    accept_start_angle_deg=4,
    seed=None

):
    """
    Used to load the environment with the wrappers combined
    """
    env = Simulator(
        seed=seed,  # random seed
        map_name=map_name,
        max_steps=max_steps,  # we don't want the gym to reset itself
        domain_rand=domain_rand,
        distortion=distortion,
        camera_width=camera_width,
        camera_height=camera_height,
        draw_curve=draw_curve,
        accept_start_angle_deg=accept_start_angle_deg
    )
    env = DtRewardWrapper(env)
    env = DiscreteWrapper(env)
    return env