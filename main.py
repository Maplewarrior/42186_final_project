import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f'Training...')

    if args.mode == 'eval':
        print(f'Evaluating...')

    if args.mode == 'sample':
        print(f'Sampling...')
    