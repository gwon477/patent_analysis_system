from jinja2 import Environment, FileSystemLoader
 
def load_jinja2_template(prompt_dir, filename):
    env = Environment(loader=FileSystemLoader(prompt_dir))
    return env.get_template(filename) 