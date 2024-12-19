import json
import openai
import argparse


def main(args):
    print(args.OPENAI_API_KEY)
    print(args.num_question)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--OPENAI_API_KEY', type=str, required=True)
    parser.add_argument('--num_question', type=int, default=3)
    args = parser.parse_args()
    main(args)