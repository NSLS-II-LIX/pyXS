import versioneer
import setuptools.command.install
import setuptools.command.build_ext
from setuptools import setup
from setuptools import Extension

try:
    from numpy import get_include as get_numpy_include
except ImportError:
    from numpy import get_numpy_include as get_numpy_include

numpy_include = get_numpy_include()


class BuildExtFirst(setuptools.command.install.install):
    def run(self):
        self.run_command("build_ext")
        return setuptools.command.install.install.run(self)


class BuildExtOnce(setuptools.command.build_ext.build_ext):
    def __init__(self, *args, **kwargs):
        # Avoiding namespace collisions...
        self.setup_build_ext_already_ran = False
        setuptools.command.build_ext.build_ext.__init__(self, *args, **kwargs)

    def run(self):
        # Only let build_ext run once
        if not self.setup_build_ext_already_ran:
            self.setup_build_ext_already_ran = True
            return setuptools.command.build_ext.build_ext.run(self)


RQ_module = Extension('_RQconv',
                      sources=['pyxs/ext/RQconv.i', 'pyxs/ext/RQconv.c'],
                      include_dirs=[numpy_include],
                      )

cmds = versioneer.get_cmdclass()
cmds['install'] = BuildExtFirst # Build first the extension and after that builds the entire package
cmds['build_ext'] = BuildExtOnce # Avoid build the extension twice

setup(
    name='pyxs',
    description="""C module for scattering pattern conversion""",
    version=versioneer.get_version(),
    cmdclass=cmds,
    author='Lin Yang, Hugo Slepicka',
    license="",
    url="https://github.com/hhslepicka/travistest",
    packages=['pyxs', 'pyxs.ext'],
    package_data={},
    py_modules=['pyxs'],
    ext_modules=[RQ_module],
    install_requires=['nose', 'numpy', 'scipy', 'matplotlib', 'Pillow', 'fabio'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.4",
    ],
)
