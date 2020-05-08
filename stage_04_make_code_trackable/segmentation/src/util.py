from dotmap import DotMap

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def dict_to_config(config_dict) -> DotMap :    
    config = DotMap(config_dict)
    return config

def json_file_to_config(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = dict_to_config(config_dict)

    return config, config_dict
