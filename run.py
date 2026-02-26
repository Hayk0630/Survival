from run_pipeline import Pipeline


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compatibility runner for Pipeline.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to train/test csv file.')
    parser.add_argument('--test', action='store_true', help='Testing mode.')
    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.run(data_path=args.data_path, test=args.test)
