import argparse
import pathlib


MODEL_MODEL_NAME_MAP = {
    'bert': ('bert-base-uncased',),
    'xlnet': ('xlnet-base-cased',)
}

TASK_NAMES = ('pdtb2_level2',)


class Command:

    here = pathlib.Path(__file__).parent
    runner = here / 'run_pdtb.py'

    def __init__(self,
                 data_dir: str,
                 model_type: str,
                 model_name_or_path: str,
                 task_name: str,
                 output_dir: str,
                 do_train: bool = True,
                 do_eval: bool = True
                 ):
        self.data_dir = data_dir
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.output_dir = output_dir
        self.do_train = do_train
        self.do_eval = do_eval

        self.num_train_epochs = 10.0
        self.per_gpu_train_batch_size = 8
        self.per_gpu_eval_batch_size = 8

    def __repr__(self) -> str:
        return f'python {str(self.runner)} ' \
               f'--data_dir {self.data_dir} ' \
               f'--model_type {self.model_type} ' \
               f'--model_name_or_path {self.model_name_or_path} ' \
               f'--task_name {self.task_name} ' \
               f'--output_dir {self.output_dir} ' \
               f'--num_train_epochs {self.num_train_epochs} '  \
               f'--per_gpu_train_batch_size {self.per_gpu_train_batch_size} ' \
               f'--per_gpu_eval_batch_size {self.per_gpu_eval_batch_size} ' \
               f'--do_train ' \
               f'--do_eval'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_root", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    args = parser.parse_args()

    data_root = pathlib.Path(args.data_root)
    output_root = pathlib.Path(args.output_root)

    commands = []
    for model_type, model_names in MODEL_MODEL_NAME_MAP.items():
        for model_name in model_names:
            for task_name in TASK_NAMES:
                for data_dir in data_root.glob("fold_*"):
                    output_dir = output_root / task_name / model_name / data_dir.name
                    commands.append(Command(
                        data_dir=str(data_dir),
                        model_type=model_type,
                        model_name_or_path=model_name,
                        task_name=task_name,
                        output_dir=str(output_dir)
                    ))

    for command in commands:
        print(command)


if __name__ == '__main__':
    main()
