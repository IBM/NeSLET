import os
import argparse


class SubmitterBase:
    def __init__(self, config):
        self.config = config

        self.base_models = {
            "blink_wiki": {
                0.01: "NeSLET_everything/scratch-shared/blink_output/p_blink_0_01/hnm_training_output_dir/pytorch_model.bin",
                0.1: "NeSLET_everything/scratch-shared/blink_output/p_blink_0_1/hnm_training_output_dir/pytorch_model.bin",
                1.0: "NeSLET_everything/scratch-shared/blink_output/p_blink_1/hnm_training_output_dir/pytorch_model.bin",
            },
            "fget_wiki_um": {
                0.01: "NeSLET_everything/scratch-shared/blink_output/p_fget_wiki_um_0_01/hnm_training_output_dir/pytorch_model.bin",
                0.1: "NeSLET_everything/scratch-shared/blink_output/p_fget_wiki_um_0_1/hnm_training_output_dir/pytorch_model.bin",
                1.0: "NeSLET_everything/scratch-shared/blink_output/p_fget_wiki_um_1/hnm_training_output_dir/pytorch_model.bin",
            },
            "fget_wiki_conll": {
                0.01: "NeSLET_everything/scratch-shared/blink_output/p_fget_wiki_conll_0_0_1/hnm_training_output_dir/pytorch_model.bin",
                0.1: "NeSLET_everything/scratch-shared/blink_output/p_fget_wiki_conll_0_1/hnm_training_output_dir/pytorch_model.bin",
                1.0: "NeSLET_everything/scratch-shared/blink_output/p_fget_wiki_conll_1/hnm_training_output_dir/pytorch_model.bin",
            },
        }

        self.data_percentage_to_num_epochs = {
            0.01: 30,
            0.1: 30,
            1.0: 4,
        }


class Submitter4(SubmitterBase):
    def __init__(self, config):
        super().__init__(config)

    def build_submission_command(self, exp_dir, dataset_name, data_percentage, num_epochs, base_model):
        # "sbatch -J job -o NeSLET_everything/scratch/log/stdout_%j -e NeSLET_everything/scratch/log/stderr_%j --export=ALL,code_dir='NeSLET_everything/NeSLET/code/blink/blink',experiment_dir='NeSLET_everything/scratch/pipeline_out',training_dataset_name='pipeline_test',percentage_training_data_to_use=100,num_vanilla_training_epochs=2,num_hnm_training_epochs=2,types_key='fine_types_id',base_model='NeSLET_everything/scratch-shared/blink_output/p_blink_0_01/hnm_training_output_dir/pytorch_model.bin' pipeline_typed_4.sh"

        command_template = "sbatch -J job -o {stdout_path} -e {stderr_path} --export=ALL,code_dir='{code_dir}',experiment_dir='{exp_dir}',training_dataset_name='{dataset_name}',percentage_training_data_to_use={data_percentage},num_vanilla_training_epochs={num_epochs},num_hnm_training_epochs={num_epochs},types_key='fine_types_id',base_model='{base_model}' {pipeline_script_path}"

        stdout_path = os.path.join(exp_dir, "stdout")
        stderr_path = os.path.join(exp_dir, "stderr")

        command = command_template.format(
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            code_dir=self.config["code_dir"],
            exp_dir=exp_dir,
            dataset_name=dataset_name,
            data_percentage=data_percentage,
            num_epochs=num_epochs,
            base_model=base_model,
            pipeline_script_path=self.config["pipeline_script"]
        )

        return command

    def submit(self):
        dataset_names = ["blink_wiki", "fget_wiki_um", "fget_wiki_conll"]
        data_percentages = [(0.01, "0_01"), (0.1, "0_1"), (1.0, "1")]

        for dataset_name in dataset_names:
            for data_percentage, data_percentage_string in data_percentages:
                exp_name = f"p_{dataset_name}_{data_percentage_string}_no_tree"
                exp_dir = os.path.join(self.config["base_dir"], exp_name)
                command = self.build_submission_command(
                    exp_dir=exp_dir,
                    dataset_name=dataset_name,
                    data_percentage=data_percentage,
                    num_epochs=self.data_percentage_to_num_epochs[data_percentage],
                    base_model=self.base_models[dataset_name][data_percentage]
                )

                os.system("mkdir {}".format(exp_dir))
                os.system(command)

                # print(command)


class Submitter5(SubmitterBase):
    def __init__(self, config):
        super().__init__(config)

    def build_submission_command(self, exp_dir, dataset_name, data_percentage, num_epochs, base_model, prob_aggregation):
        # sbatch -J job -o zel_everything/scratch/log/stdout_%j -e zel_everything/scratch/log/stderr_%j --export=ALL,code_dir='zel_everything/NeSLET/code/blink/blink',experiment_dir='zel_everything/scratch/pipeline_out',training_dataset_name='pipeline_test',percentage_training_data_to_use=100,num_vanilla_training_epochs=2,num_hnm_training_epochs=2,types_key='fine_types_id',base_model='zel_everything/scratch-shared/blink_output/p_blink_0_01/hnm_training_output_dir/pytorch_model.bin',prob_aggregation='godel' pipeline_typed_5.sh

        command_template = "sbatch -J job -o {stdout_path} -e {stderr_path} --export=ALL,code_dir='{code_dir}',experiment_dir='{exp_dir}',training_dataset_name='{dataset_name}',percentage_training_data_to_use={data_percentage},num_vanilla_training_epochs={num_epochs},num_hnm_training_epochs={num_epochs},types_key='fine_types_id',base_model='{base_model}',prob_aggregation='{prob_aggregation}' {pipeline_script_path}"

        stdout_path = os.path.join(exp_dir, "stdout")
        stderr_path = os.path.join(exp_dir, "stderr")

        command = command_template.format(
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            code_dir=self.config["code_dir"],
            exp_dir=exp_dir,
            dataset_name=dataset_name,
            data_percentage=data_percentage,
            num_epochs=num_epochs,
            base_model=base_model,
            pipeline_script_path=self.config["pipeline_script"],
            prob_aggregation=prob_aggregation
        )

        return command

    def submit(self):
        dataset_names = ["blink_wiki", "fget_wiki_um", "fget_wiki_conll"]
        data_percentages = [(0.01, "0_01"), (0.1, "0_1"), (1.0, "1")]
        prob_aggregations = ["godel", "lukasiewicz"]

        for dataset_name in dataset_names:
            for data_percentage, data_percentage_string in data_percentages:
                for prob_aggregation in prob_aggregations:
                    exp_name = f"p_{dataset_name}_{data_percentage_string}_{prob_aggregation}"
                    exp_dir = os.path.join(self.config["base_dir"], exp_name)
                    command = self.build_submission_command(
                        exp_dir=exp_dir,
                        dataset_name=dataset_name,
                        data_percentage=data_percentage,
                        num_epochs=self.data_percentage_to_num_epochs[data_percentage],
                        base_model=self.base_models[dataset_name][data_percentage],
                        prob_aggregation=prob_aggregation
                    )

                    os.system("mkdir {}".format(exp_dir))
                    os.system(command)

                    # print(command)


if __name__ == "__main__":
    """
    python zel_everything/NeSLET/code/blink/automation/pipeline_typed_job_submitter.py --base_dir zel_everything/scratch-shared/blink_output/hard_param_sharing --pipeline_script zel_everything/NeSLET/code/blink/automation/run_pipeline_typed_4.sh --code_dir zel_everything/NeSLET/code/blink/blink --type_model 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--pipeline_script", type=str)
    parser.add_argument("--code_dir", type=str)
    parser.add_argument("--type_model", type=int)

    command_line_args = parser.parse_args()
    command_line_args = command_line_args.__dict__

    if command_line_args["type_model"] == 4:
        submitter = Submitter4(command_line_args)
    elif command_line_args["type_model"] == 5:
        submitter = Submitter5(command_line_args)
    else:
        raise ValueError

    submitter.submit()








