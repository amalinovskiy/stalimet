from pybuilder.core import use_plugin, init

use_plugin('python.core')
use_plugin('python.install_dependencies')
use_plugin('python.distutils')

default_task = 'publish'


@init
def initialize(project):
    project.name = 'stalimet'
    project.version = '0.1.0-SNAPSHOT'
    project.depends_on('nltk')
    project.set_property('name', project.name)
    project.set_property('version', project.version)
    project.set_property('dir_dist_scripts', 'scripts')
    project.set_property('dir_dist', '$dir_target/dist/$name-$version')