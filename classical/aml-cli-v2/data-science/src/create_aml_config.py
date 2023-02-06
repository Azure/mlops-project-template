import argparse
import utils 

def parse_args():
    '''Parse input arguments.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--aml_config_path", type=str, help="Path to resulting AML config file.")

    args = parser.parse_args()

    return args

def main(args):
    aml_config = f"""
    {{
    "subscription_id": "{utils.fs_config.get('subscription_id')}",
    "resource_group": "{utils.fs_config.get('resource_group')}",
    "workspace_name":"{utils.fs_config.get('workspace_name')}"
    }}
    """

    with open(args.config_path, "w") as file:
        file.write(aml_config)

if __name__ == '__main__':
    args = parse_args()

    main(args)