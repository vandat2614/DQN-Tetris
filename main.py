import argparse
import yaml
import torch


from src import train, test, NeuralNetwork
from tetris import TetrisEnv

def parse_args():
    parser = argparse.ArgumentParser(description='DQN Training, Evaluate, and Testing')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'test'],
                        help='train, evaluate or test the agent')
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help='path to config file')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='path to model weights for testing')
    return parser.parse_args()

def setup(config_path : str, weights_path : str = None):
    with open(config_path ,'r') as file:
        config = yaml.safe_load(file)
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if weights_path is not None:
        config['env']['params']['sound'] = True
        config['env']['params']['render_mode'] = 'human'
    else:
        config['env']['params']['render_mode'] = 'rgb_array'

    env = TetrisEnv(**config['env']['params'])
    model = NeuralNetwork.from_config(config=config['model'], 
                                      input_size=env.observation_space.shape[0],
                                      output_size=env.action_space.n)

    if weights_path is not None:
        model.load(path=weights_path)


    return env, model, config, device

def main():
    args = parse_args()
    env, model, config, device = setup(args.config_path, args.weights_path)

    if args.mode == 'train':
        train(env, model, config['train'], device)
    elif args.mode == 'test':
        test(env, model, device)
    # elif args.mode == 'eval':
    #     evaluate(env, model, config['evaluate'], device)

if __name__ == '__main__':
    main()