"""
This module saves and writes configuration files for the configuration of the
vision system.
"""
from configparser import SafeConfigParser

LIST_OPTIONS = ["maxs", "mins"] #options in config to be interpreted as list

def list_to_string(lst):
    """
    convert a list to string format for config files
    """
    return ",".join(map(str, lst))

class ConversationConfig:
    """
    This class handles the configuration file parsing and saving
    """
    def __init__(self, path):
        self.parser = SafeConfigParser()
        self.parser_config_path = ""
        self.default_configuration(path)

    def save_config(self, config, section="default"):
        """
        saves given config section map dict to disk
        """
        for key, val in config.items():
            val_str = val
            if key in LIST_OPTIONS:
                val_str = ",".join(map(str, val))
            self.parser.set(section, key, val_str)
        print("Saving config to" + self.parser_config_path + ":")
        print(self.get_section_map(section))
        with open(self.parser_config_path, 'w') as configfile:
            self.parser.write(configfile)

    def default_configuration(self, path):
        """
        load the default configuration
        """
        self.load_config(path)
        default_config = self.get_section_map("default")
        self.config = default_config
        return default_config

    def get_section_map(self, section):
        """
        get section map dictionary of parsed config
        """
        dict1 = {}
        options = self.parser.options(section)
        for option in options:
            try:
                dict1[option] = self.parser.get(section, option)
                if dict1[option] == -1:
                    DebugPrint("skip: %s" % option)
                if option in LIST_OPTIONS:
                    dict1[option] = dict1[option].split(",")
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1

    def load_config(self, path):
        """
        load configuration from disk
        """
        print("Loading config from " + path + ".")
        self.parser_config_path = path
        self.parser.read(path)
