"""Dataset configurations used by the current dual-decoder ACT training jobs."""

SIM_TASK_CONFIGS = {
    'sim_Peach_in_bowl': {
        'dataset_dir': '/mnt/additional/Data/DexMimic_Data/Robot_data_ACT/Peach_in_bowl',
        'num_episodes': 300,
        'episode_len': 250,
        'camera_names': ['front_RGB'],
        'state_dim': 16,
    },
    'sim_Insert_cup': {
        'dataset_dir': '/mnt/additional/Data/DexMimic_Data/Robot_data_ACT/Insert_cup/inspire',
        'num_episodes': 300,
        'episode_len': 150,
        'camera_names': ['front_RGB'],
        'state_dim': 12,
    },
    'sim_Peach_in_bowl_inspire': {
        'dataset_dir': '/mnt/additional/Data/DexMimic_Data/Robot_data_ACT/Peach_in_bowl/inspire',
        'num_episodes': 299,
        'episode_len': 130,
        'camera_names': ['front_RGB'],
        'state_dim': 12,
    },
    'sim_Peach_in_bowl_inspire_human_robot': {
        'dataset_sources': [
            {
                'name': 'robot',
                'dataset_dir': (
                    '/mnt/additional/Data/DexMimic_Data/Robot_data_ACT/'
                    'Peach_in_bowl/inspire'
                ),
                'num_episodes': 299,
            },
            {
                'name': 'human',
                'dataset_dir': (
                    '/mnt/additional/Data/DexMimic_Data/Human_data/'
                    'Peach_in_bowl'
                ),
                'num_episodes': 395,
            },
        ],
        'episode_len': 182,
        'camera_names': ['front_RGB'],
        'state_dim': 12,
    },
}
