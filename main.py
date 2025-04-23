from pettingzoo.mpe import simple_spread_v3
from PickUpDroppOffSimpleSpread import PickUpDropOffSimpleSpread
from mappo import MAPPO

def main():
    base_env = simple_spread_v3.env(N=3, local_ratio=0.5, max_cycles=25)
    wrapped_env = PickUpDropOffSimpleSpread(base_env, num_tasks=1)
    pol = MAPPO(wrapped_env)

    for episode in range(2000):
        memory = pol.collect_trajectory(horizon=100)
        pol.update(memory)
        if episode % 100 == 0:
            print(f'Episode {episode} complete')


main()
