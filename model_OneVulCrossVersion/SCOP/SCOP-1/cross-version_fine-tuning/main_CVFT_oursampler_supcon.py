import wandb
import configparser
import subprocess
import shutil

# Sweep配置
# sweep_config = {
#     'method': 'grid',  # 或者bayes, random
#     'metric': {
#         'name': 'validation_accuracy',
#         'goal': 'maximize'
#     },
#     'parameters': {
#         'learning_rate': {
#             'values': [0.001, 0.005, 0.0001, 0.0005]
#         },
#         'train_batch_size': {
#             'values': [16, 24, 32, 48]
#         },
#         'epochs': {
#             'values': [2, 5, 10, 20, 30]
#         }
#     },
#     'early_terminate': {
#         'type': 'hyperband',
#         'min_iter': 3,
#         'eta': 2
#     }
# }

sweep_config = {
    'method': 'random',  # 或者bayes, random
    'metric': {
        'name': 'validation_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-6, 1e-5]
        },
        'train_batch_size': {
            'values': [24, 32, 48]
        },
        'epochs': {
            'values': [5, 10, 14]
        },
        'sample_per_class_per_version': {
            'values': [2, 4, 8, 16]
        },
        'code_length': {
            'values': [256, 512]
        }
    }
}



def modify_and_run(cfg_file_path, script_path, learning_rate, train_batch_size, epochs):
    config = configparser.ConfigParser()
    config.read(cfg_file_path)
    # 修改配置文件
    config.set('parameters_section', 'learning_rate', str(learning_rate))
    config.set('parameters_section', 'train_batch_size', str(train_batch_size))
    config.set('parameters_section', 'epochs', str(epochs))  # 添加这行代码
    with open(cfg_file_path, 'w') as configfile:
        config.write(configfile)
    
    # 运行训练脚本
    subprocess.run(['python', script_path], check=True)


def sweep_train():
    # 创建sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project="CLPR_ourSampler_SupCon_CVFT_0308")
    
    def train():
        # 初始化W&B
        wandb.init()
        learning_rate = wandb.config.learning_rate
        train_batch_size = wandb.config.train_batch_size
        epochs = wandb.config.epochs

        # 定义配置文件和脚本路径
        cfg_file_path = '/home/chen/workspace/codeproject/CL4acrossVersionSC/model_OneVulCrossVersion/ourCLPT/ourSampler_SupCon/cross-version_fine-tuning/devCLPT_ourSampler_wandb.cfg'
        script_path = '/home/chen/workspace/codeproject/CL4acrossVersionSC/model_OneVulCrossVersion/ourCLPT/ourSampler_SupCon/cross-version_fine-tuning/CLPT_ALL_Sampler_SupCon_wandb.py'

        # 备份原始配置文件
        cfg_backup_path = cfg_file_path + '.backup'
        shutil.copyfile(cfg_file_path, cfg_backup_path)

        # 修改并运行训练脚本
        modify_and_run(cfg_file_path, script_path, learning_rate, train_batch_size, epochs)

        # 恢复原始配置文件
        shutil.move(cfg_backup_path, cfg_file_path)
        
        wandb.finish()

    # 启动sweep
    wandb.agent(sweep_id, function=train)

if __name__ == '__main__':
    sweep_train()
